# LiteArm 真机硬件路径

本页说明 LiteArm / LiteGrip 真机工作站的适配：可单臂或双臂，每只 active 臂
可带夹爪或不带夹爪，相机可为头部/腕部中任意已配置子集。它与 RoboDojo 的
LiteArm 仿真适配相互独立：真机路径不导入、不启动 RoboDojo，不读取其任务、
布局、资产或 `.cache`，也不写 `eval_result`。

## 状态概览

| 项目 | 状态 |
| --- | --- |
| 离线端到端：runner → `robo` → toolkit → localhost RPC → 假设备 → 记录 | 已实现并测试 |
| adapter、配置/标定校验、FK/IK/TCP、规划、近似碰撞 | 已实现并离线测试 |
| RPC 去重、独占会话、取消、超时、断连、故障状态 | 已实现，并用真实 localhost socket 测试 |
| RGB-D / 多视角感知 | 已实现；用已知几何测试；SAM3 HTTP endpoint `http://192.168.55.1:8790` 已于 2026-11-05 验证 |
| 厂商 SDK 封装 | 在设备环境用 `litearm.testing.FakeTransport` 和 mock 夹爪测试 |
| 真机运动、急停保持/失能、固件抢占、标定精度 | **本次未在真机执行，需现场验收** |
| 真实相机采集（OpenCV / RealSense） | 已实现；本次未连接相机 |

## 运行架构

两个进程：

1. **Runner / agent**（`pyrualean-hw-agent` 环境）运行模型生成的 Python，
   通过正常 `robo` API 调用 `LitearmRealRobot` → `HardwareToolkit`；
   toolkit 负责规划、感知、记录证据，并作为 RPC 客户端。
2. **硬件服务**（`litearm-hw-device` 环境）独占已配置的机械臂、LiteGrip
   和相机。它再次校验轨迹，执行同步关节段，采集实测状态，并管理会话、
   停止和故障状态机。

服务默认只监听 `127.0.0.1`。协议为 TCP 上的逐行 JSON；首版没有 TLS 和认证，
若跨机部署必须用防火墙或安全隧道保护。一个 runner 会话独占设备。

```
策略代码 -> LitearmRealRobot -> HardwareToolkit -> HardwareClient
                                                    |
                                        localhost TCP JSON
                                                    |
HardwareDaemon -> 规划复检 -> LiteArmStation -> LiteArm SDK
                                             -> LiteGrip SDK
                                             -> CameraManager
```

所有硬件 SDK、相机 SDK 只在设备创建函数内部导入。导入
`pyrualean.hardware` 或 `pyrualean.litearm_real` 不会连接设备、启动线程、
加载 GPU 模型或要求安装硬件 SDK。

## 已核对的 SDK 接口和版本

- `litearm-python` **2.1.0**，本地 `/home/hzh/Code/litearm-python`，commit
  `5643e47`（git describe `v1.3.0-3-g5643e47`；包内 `__version__` 为 2.1.0）。
  它不是 PyPI 上的 1.3.0。
- `litegrip-python` **2.2.0**，本地 `/home/hzh/Code/NEXTFORM/litegrip-python`，
  commit `6b5fdce`（git describe `v1.2.1-2-g6b5fdce`；源码树使用
  semantic-release 占位版本 `0.0.0`，已核对的安装报告为 2.2.0）。它不是
  PyPI 上的 1.2.1。

LiteArm 使用：

- `Arm.connect()`、`get_state(refresh=True)`（`Msg.value`）、`get_tcp()`、
  `get_tool_tcp()`、`set_tcp_offset(xyz, rpy)`；
- `movej_sync(q, speed)` 阻塞执行，`speed` 为 0..1；到位由实测状态判断。
  不用 `movej` 做线段，因为各轴独立 S 曲线使中间末端轨迹不可预测。
- `emergency_stop()` 是固件命令；`disable()`、`park()` 是另外的动作，不能
  当作受控停止。
- 不暴露 `move_js`：它需要 ≥10 Hz 重发和 0.1 s 看门狗；首版没有固定频率
  伺服调度、保活和停止机制。
- `get_tcp()` 返回法兰位姿；`get_tool_tcp()` 套用宿主侧 TCP 偏移。真机路径
  用实测关节 + 标定计算 TCP，并把固件法兰位姿作为交叉核对。

LiteGrip 使用：

- `LiteGrip(channel, can_id)`、`connect()`、`load_calibration(path=...)`、
  `get_state(wait=False)` 的 `position_mm`、`force_n`、`fresh`、
  `data_age_s`、`error_code`；
- `grasp(force_n, hold_s=0)` 会阻塞保持，由另一线程 `release()` 结束；
  每个夹爪只有一个持久 grasp 线程，重复 KEEP/close 不会重启夹取；
- 开启/部分开启用 `move_at_speed(target_mm, speed_mm_s)`；急停路径用
  `request_release(abort=True)`；`stop()` 只发一帧零力矩，不是安全级停止。

SDK 使用固定版本 wheel，不使用 editable 安装。用
`scripts/build_litearm_device_wheels.sh` 构建；LiteGrip 仓库的
semantic-release 占位版本在临时构建副本里替换成已核对的发布版本，不修改
输入 checkout。

## 安装

### Agent 环境

```bash
conda env create -f environments/litearm-agent.yml
conda activate pyrualean-hw-agent
python -m pip install -e .
```

Agent 环境不装硬件 SDK。SAM3 部署在 Jetson Thor，agent 只需要标准库 HTTP
客户端和站点文件中的 endpoint；离线测试用脚本化/HTTP segmenter，不依赖
`torch` 或 `sam3` 包。

### Jetson Thor 上的 SAM3

SAM3 作为 HTTP 服务部署在 Jetson Thor。在 Thor 上（`sam3` 包、torch 和
checkpoint 所在的机器）：

```bash
python src/pyrualean/hosts/sam3_http.py \
  --host 0.0.0.0 --port 8790 \
  --checkpoint /home/nextform/PyRUA/PyRUA-Lean/integrations/xpolicylab/PyRUA_Lean/checkpoints/sam3.pt
```

在 agent 机器上检查（设备网是直连，绕过 HTTP 代理）：

```bash
curl --noproxy '*' http://192.168.55.1:8790/health
# {"ok": true, "served": 0}
```

Agent 的站点文件指向它：

```json
"perception": {
  "sam3": {
    "mode": "http",
    "endpoint": "http://192.168.55.1:8790",
    "timeout_s": 60.0
  }
}
```

`192.168.55.1` 是工作站 handoff 中记录的 Thor USB 设备网地址；网络变化时改这里。
2026-11-05 已用 `/health` 和一次点提示分割验证该地址可连通。本版服务没有认证，
请限制在可信机器人网络或隧道内。`mode="local"` + checkpoint 在本地 GPU 机器上
仍然可用，但默认真机示例使用 Thor endpoint。

### 设备环境

```bash
conda env create -f environments/litearm-device.yml
conda activate litearm-hw-device
LITEARM_SDK_SRC=/path/to/litearm-python \
LITEGRIP_SDK_SRC=/home/hzh/Code/NEXTFORM/litegrip-python \
scripts/build_litearm_device_wheels.sh
python -m pip install --no-index --find-links wheelhouse \
    litearm-python==2.1.0 litegrip-python==2.2.0
python -m pip install --no-deps -e .
```

驱动真机的机器不要用厂商 SDK 的 editable 安装。

## 站点配置与标定

从 `configs/litearm_real.example.json` 复制出本机站点文件，从
`configs/calibration/litearm_real.example.json` 复制出标定文件。配置文件中的
相对路径以配置文件所在目录为基准。真实站点文件、标定、凭据和结果均被
`.gitignore` 忽略，示例保留。

关键配置：

- `service`：地址、端口、连接/请求超时、心跳周期/超时、最大时钟偏差。
  默认 localhost。
- `execution`：`mode`（`read-only`、`offline-fake`、`live`）、可选
  `active_arms`（默认 `["left", "right"]`；只装一只臂时设为 `["left"]` 或
  `["right"]`，此时另一只臂的 `devices` 条目和腕部相机可以省略，服务不会
  打开或控制它）、速度比例、
  配置的关节/TCP 上限、段长、稳定容差和命令超时。SDK 只暴露速度比例；
  加速度由固件负责，本层**不声称**已强制动态加速度限制。
- `devices`：每臂串口和最低固件、每夹爪 CAN 通道/ID、可选标定文件、夹持力。
  该 active 臂没有 LiteGrip 时，`gripper` 可为 `null` 或整个省略：服务只
  打开、读取和停止机械臂，`state()` 的 `gripper`/`width_m` 为 `None`，
  OPEN/CLOSE 和 `place_on` 会明确拒绝，绝不会伪造夹爪。
  省略 `calibration_path` 时使用 SDK 随包出厂标定；`can1` 夹爪需要自己的
  文件，因为出厂文件声明的是 `can0`。
- `cameras`：驱动（`opencv`、`realsense`；`fake` 仅允许 offline-fake）、
  索引/序列号、分辨率、帧率、是否深度。`opencv` 还接受 `device`（显式路径，
  如 `/dev/video4`，用 `cv2.CAP_V4L2` 打开，优先于 `index`）和 `fourcc`
  （如 `"YUYV"`，在分辨率/帧率协商前设置）。整数索引无法寻址所有 V4L2
  节点；`device`/`fourcc` 对非 opencv 驱动会报错，opencv 相机必须有索引
  或设备路径。
- `perception`：SAM3 模式（Jetson Thor 用 `http`，也可用 `local`）、endpoint
  和超时、可选本地 checkpoint/GPU、体素、最大物体高度、最小有效深度比例，
  以及 `primary_camera`（默认 `"cam_head"`，必须是已配置相机）。runner 的
  `locate`、`segment`、`back_project`、`image`、`camera_meta` 在调用方省略
  `camera` 时使用它，因此只有腕部相机的站点也能直接 `locate(prompt)`。
- `snapshot`：最大数据年龄、相机/关节同步限制、刷新间隔、历史数。
- `results`：结果目录、设备动作预算、墙钟预算。
- `safety`：停止超时、静止确认阈值、home 抬升高度、故障后操作员复位策略。
- `fake`：仅 offline-fake 使用的确定性参数；live 忽略。

`execution.mode=live` 只有在所选机械臂和相机的标定 `"executable": true`
且质量状态不是示例/未验证时才允许。示例标定故意为 `executable: false`。

### 标定内容

- 世界原点：靠近机器人一侧桌沿中心；+x 向右、+y 向桌内、+z 向上，桌面
  z=0。
- 每臂完整基座位姿、法兰到 TCP 变换、关节映射（模型关节、SDK 索引、符号、
  零位偏移）、实测关节限位、安全 home。
- LiteGrip 实测闭合/张开宽度（归一化开口 =
  `(width - closed)/(open - closed)`）、夹持力、最小夹持宽度。
- 每个相机的 camera-to-world（头部）或 camera-to-tool（腕部）变换、内参、
  畸变、分辨率、是否有深度。
- 桌面范围和固定障碍。
- 来源、时间、版本和质量。

所有矩阵校验形状、有限值、旋转正交且行列式 +1。关节映射必须把模型关节
0..6 和 SDK 索引 0..6 各用一次，符号 ±1。限位必须落在打包 URDF 内。安全
home 必须在限位内。

标定必须**恰好**描述站点实际设备：`arms` 的键集合必须等于
`execution.active_arms`，`cameras` 的键集合必须等于配置的 `cameras`。
单臂/仅腕部相机的站点因此使用只含实际条目的裁剪标定文件；加载器既不要求
也不伪造缺失设备的条目。随包的两臂/三相机示例对默认配置仍然有效。
`load_calibration` 也接受显式 `arms=`/`cameras=` 选择器（默认全部），并执行
同样的精确集合校验。

### 人工测量步骤

`tools/litearm_real/calibrate.py` 只校验并写入测量文件，**不会**移动机械臂；
传 `--allow-motion` 会明确拒绝，而不是偷偷做运动采集。每项应人工测量并
记录方法：

1. **基座**：在已知关节构型下测基座原点/轴向，写 base-to-world 4x4。用多个
   位姿比较 `get_tcp()` 与 FK；观测构建器对超过 20 mm 的偏差报错。
2. **TCP**：测量抓取点相对法兰的位置/姿态（含 LiteGrip 安装），写
   `flange_to_tcp`。在 SDK 设同一偏移，用多个位姿比较 `get_tool_tcp()`。
3. **关节**：逐轴核对 SDK 顺序、方向、零位；手动或操作员低速工具逐轴运动，
   记录映射和实际软限位。不要猜符号。
4. **夹爪**：LiteGrip SDK 自带 `src/litegrip/factory_calibration.json`
   （`calibrated: true`、`channel: can0`），所以 `can0` 夹爪不需要额外文件，
   不带参数的 `load_calibration()` 会用它。SDK 故意不会在 `can1` 上隐式复用
   这份文件；请对该夹爪运行厂商的 `zero()`/`calibrate_guided()`/
   `calibrate_manual()`，并把 `devices.<arm>.gripper.calibration_path` 指向
   生成的 `~/.litegrip/<channel>_calibration.json`。用卡尺测闭合/全开宽度，
   验证命令停止在标定行程内。力是 SDK 名义换算，不是标定力，不能用于安全
   决策。下面的 `calibrate.py` 只写**站点**标定（base/TCP/相机），不生成
   这个厂商夹爪文件。
5. **相机**：用标定板测内参/畸变；头部测 camera-to-world，腕部测
   camera-to-tool。分辨率必须一致。RealSense 驱动会拒绝与标定差异超过 2%
   （主点 4 px）的设备内参。RealSense 彩色流请求 `rs.format.bgr8`，返回前
   转为 RGB：D435i 固件 5.17.0.9 在 640x480@30 下 `rgb8` 彩色流会超时，
   `bgr8` 可稳定同时输出彩色和深度。若 OpenCV 相机无法用整数索引打开
   V4L2 节点，设置 `cameras.<name>.device` 为路径（如 `/dev/video4`），
   必要时设 `fourcc`（如 `"YUYV"`）；驱动用 `cv2.CAP_V4L2` 打开路径，并在
   宽/高/帧率之前设置 FourCC。
6. **Home**：选一个无碰撞关节构型作为 `safe_home_rad`，用规划器验证抬升和
   返回路径。它不是 SDK 零位。

把工作副本中的示例值全部替换为实测值，把 `source` 改为真实测量记录，
把 `quality.status` 改为已测量状态，然后写入：

```bash
python tools/litearm_real/calibrate.py \
  --from /abs/path/to/measured_calibration.json \
  --out configs/calibration/litearm_real.json \
  --measurement-quality measured \
  --mark-executable
```

输入文件仍带 `example`、`template` 或 `unverified` 标记时，工具拒绝标记为
可执行。只有全部测量并复核后才设 `executable=true`。未验证/示例标定会被
live 拒绝。

## 启动服务

```bash
LITEARM_HW_CONFIG=/abs/path/station.json \
LITEARM_HW_MODE=live \
LITEARM_HW_LOG=/abs/path/service.log \
scripts/litearm_hardware_service.sh
```

离线验证用 offline 配置和 `LITEARM_HW_MODE=offline-fake`。服务：

- 不自动 enable、reset、清故障或回零；
- live 下若臂或夹爪未使能/有故障则拒绝启动（由操作员检查后在服务外处理）；
- 同一时刻只允许一个 runner 会话；
- read-only 允许快照，但拒绝运动。

## 运行 episode

```bash
LITEARM_HW_CONFIG=/abs/path/station.json \
LITEARM_HW_INSTRUCTION='把桌上的红色积木放到托盘里' \
LITEARM_HW_MODE=live \
LITEARM_HW_BUDGET_S=1200 \
scripts/litearm_hardware_play.sh
```

等价直接命令：

```bash
pyrualean play \
  --robot litearm_real \
  --arm cells \
  --primitives no-vla \
  --hardware-config /abs/path/station.json \
  --task pick_place \
  --instruction "把桌上的红色积木放到托盘里" \
  --out /abs/path/results_hw/run_id
```

只有显式 `--robot litearm_real` 才选择真机 adapter；没有环境变量会偷偷改变
默认 litearm 仿真后端。`read-only` 仍连接并感知，但拒绝运动；
`offline-fake` 使用合成设备和相机，绝不会因真实驱动失败而自动回退；
`live` 要求所选 active 臂和已配置相机的标定可执行且质量已测量。

## 停止语义

- **普通取消**：在同步段边界后用新鲜状态读回确认，臂保持实测位姿，服务进入
  `stopped`。之后再发轨迹必须重启服务。
- **超时/心跳丢失/显式急停**：服务调用固件急停并进入 `fault`。结果记录停止
  是否得到确认；未确认时报告 `stop_unconfirmed`，不能声称停止成功。
- 服务不把 `disable()` 或 `park()` 当作受控停止。
- 故障不会自动清除；重连不会恢复运动。
- 急停在真机上的保持/失能行为、能否抢占阻塞段，均为**现场验收项**。

## 结果与诊断

episode 目录包含：

- `prompt.txt`、`calls.jsonl`、`tool_turns.json`、`result.json`；
- `steps/NNNNNN/`：`snapshot.json`、`observation.json`、`camera_meta.json`、
  `device.json`、`artifacts/...`（PNG/NumPy）；
- `episode.json`：停止证据和成功语义；
- `hardware_host.json`、`card/`（图片和 `card.json`）。

`State.step` 是观测快照编号；`steps_used`/`env_steps` 是已执行轨迹段或夹爪
动作数。`environment_success` 为 false，`environment_success_known` 为 false，
`task_success` 为 `not_measured`。模型 `finish(status="success")` 单独记录，
不是任务成功判定。

## 诊断

- `tools/litearm_real/smoke.py`：只读检查握手、哈希、时钟、快照、TCP 和停止
  证据。`--allow-motion` 可做小运动/单物体抓放；本次未在真机执行该路径。
- 服务日志记录状态读失败、故障原因和停止确认。
- 检查 `steps/*/camera_meta.json` 与 `device.json` 的一致性。
- `step=-1` 的 `camera_meta` 与 `image` 来自同一固定快照；之后刷新状态会创建
  新快照。

## 测试

Agent 环境（无硬件 SDK、无本地 SAM3/torch）：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_litearm.py tests/test_litearm_real.py \
  tests/test_litearm_real_config.py tests/test_litearm_real_geometry.py \
  tests/test_litearm_real_perception.py tests/test_litearm_real_execution.py \
  tests/test_litearm_real_cameras.py tests/test_cli.py -q
```

设备环境：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_litearm_real_devices.py -q
```

设备测试使用 `litearm.testing.FakeTransport` 和 mock LiteGrip，不是物理验证。
执行测试使用真实 localhost TCP 和合成站点，不是只 mock 客户端方法。

## 已知限制 / 待现场验证

- 所有真实运动、夹持力、急停、固件行为、线缆时序和标定精度都待现场验证。
- 碰撞几何是保守的胶囊/盒体近似，不是 mesh 精确碰撞，也不构成避障认证。
- SDK 只暴露速度比例；段长和超时是运行护栏，不是动态极限证明。
- 首版服务没有 TLS/认证；请限制在本机或受保护网络。
- SAM3 预期部署在 Jetson Thor；示例指向上面已验证的地址，但 HTTP 服务必须
  正在运行。不可达时感知调用明确失败；本地 `sam3`/checkpoint 仍作为可选后备。
- RealSense 需要 `pyrealsense2` 和匹配固件；本次未连接相机。
- 示例配置/标定不可执行；未测量前不要设 `executable=true`。

## API reference

<!-- api-reference: litearm_real primitives=no-vla -->
