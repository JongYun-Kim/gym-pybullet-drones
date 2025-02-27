""" VisionLandingAviary.py
이 코드는 gym-pybullet-drones의 BaseSingleAgentAviary를 확장하여
vision 기반 드론 자동 착륙 학습을 위한 Gym 환경의 구현

주요 특징:
    - 드론 모델: CF2X (craziflie 2.0, x configuration)
      - Total Joints: 5
        - Link Index: 0, Name: prop0_link,          Position: ( 0.02800,   0.02803,   0.01349)
        - Link Index: 1, Name: prop1_link,          Position: (-0.02800,   0.02803,   0.01349)
        - Link Index: 2, Name: prop2_link,          Position: (-0.02800,  -0.02780,   0.01349)
        - Link Index: 3, Name: prop3_link,          Position: ( 0.02800,  -0.02780,   0.01349)
        - Link Index: 4, Name: center_of_mass_link, Position: (-3.300e-07, 2.831e-05, 0.0134901)
      - 평면에 착륙하면 z = 0.0135 (base==self.pos)
      - 패드 착륙
        - 이론상: z = 0.2860 == 0.2725 + 0.0135 (heliport_base + drone_base_height)
        - 실험서: z = 0.28754655
    - 헬리패드: 움직이는 착륙 패드 (simple car + helipad texture from parsed_pad.urdf in assets dir)
    - Body ID: 0, Name: plane  as idk yet
    - Body ID: 1, Name: cf2    as self.DRONE_IDS
    - Body ID: 2, Name: car    as self.landing_pad_id
    - Observation: 드론 카메라에서 촬영한 RGB 이미지(알파 채널 제외)를 4 프레임 stack
    - Action: ActionType.RPM or VEL
    - Reward: Visibility + Safety (vertical vel) + Landing/Crashing + Horizontal distance
    - 기록: 환경 외부 카메라 기록(BaseAviary의 record 옵션) 외에도, onboard 카메라 이미지가 PNG로 저장되며, 후에 ffmpeg로 동영상 변환 가능.
"""
# TODOs:
# - [o] rgb to grey scale
# - [o] Check observation stack interval
#   - 30 Hz 로 이미지 촬영; 4 프레임 크기의 buffer를 새로운 프레임으로 업데이트
#   - aggregate_phy_steps=10 을 하면 freq=240Hz에서 알아서 매 observation 마다 RL-action 을 할 기회를 줌 (빠른 simulation)
# - [o] Replace the reward function with Pawel's
# - [o] Check done condition
# - [o] drone state를 제대로 obs 넣어줘야 함! (현재는 world 좌표에서 인듯; linear vel, quat 넣으면 될 듯)
# - [o] Check the curricula
# - [o] Implement the curriculum learning
# - [o] Update action space
# - [o] Randomize landing pad direction (yaw)
# - [o] Randomize drone initial position and orientation
#   - [ ] LOS check (vision 범위에서 시작해야함)
# - [ ] Laters:
#   - [ ] Make a configuration for the desired z velocity: @ the def of 'self.SPEED_LIMIT' in BaseSingleAgentAviary
#   - [ ] Randomize landing pad texture
#   - [o] Check getDroneImage method (rotation wise...)
# (4) _computeDone 체크 하기! contact 를 체크 해서 curriculum learning 에 통합 해야함.
import os
import numpy as np
import pybullet as p
from gym import spaces
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import BaseSingleAgentAviary, ObservationType, ActionType
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.BaseAviary import ImageType  # onboard 이미지 저장에 사용
from gym_pybullet_drones.utils.utils import rgb2gray
import subprocess
from scipy.spatial.transform import Rotation
from gym_pybullet_drones.utils.utils_geometry import project_point_on_pad_plane, inside_pad_box, order_points_convex_polygon, polygons_intersect_2d, line_plane_intersection
from gym_pybullet_drones.utils.utils_ffmpeg import convert_images_to_video


class VisionLandingAviary(BaseSingleAgentAviary):
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 freq: int = 300,
                 aggregate_phy_steps: int = 10,
                 gui: bool = False,
                 record: bool = False,
                 obs: ObservationType = ObservationType.BW,
                 act: ActionType = ActionType.VEL,
                 channel_first: bool = True,     # nn.Conv2d(): must: channel_first=True
                 stack_size: int = 4,            # Number of frames to stack
                 fov: float = 80.0,              # Field of view for the drone camera (in degrees!)
                 img_res: np.ndarray = np.array([84, 84]),  # original: np.array([64, 48])
                 img_fps: int = 30,
                 episode_len_sec: float = 30.0,   # 에피소드 길이 (초)
                 include_drone_state: bool = True,
                 difficulty: int = 4,
                 curriculum_configs: dict = None,
                 # **kwargs, # Enable this ONLY IF you need additional arguments as a workaround (e.g. curriculum plans)
                 ):
        # Curriculum learning settings
        self.difficulty = difficulty
        if curriculum_configs is not None:
            assert "metric" in curriculum_configs, "Metric should be provided in the curriculum_configs."
            assert "plans" in curriculum_configs, "Plans should be provided in the curriculum_configs."

        assert include_drone_state, "Currently, include_drone_state == False is not supported."
        self.stack_size = stack_size  # used in _observationSpace(), which is called in super().__init__()
        this_file_dir = os.path.dirname(os.path.realpath(__file__))
        self.assets_path = os.path.join(this_file_dir, "../../assets")
        self.fov = fov
        self.channel_first = channel_first

        # 그레이 스케일 사용 여부 (Complicated; but tried to maintain backward compatibility)
        self.use_gray_scale = True if obs == ObservationType.BW else False
        obs = ObservationType.RGB if obs == ObservationType.BW else obs

        # 착륙 패드 관련 파라미터 초기화 (IMG_RES 와 무관 하므로 먼저 호출 가능)
        self.pad_link_center_idx = None
        self._resetLandingPad()
        self.pad_height = 0.2725  # 착륙 패드 높이 (착륙후 drone pos 는 이거 보다 높음)

        # 드론 초기화
        self.user_xyz = initial_xyzs
        self.user_rpy = initial_rpys
        initial_xyzs, initial_rpys = self._get_random_drone_pose(self.user_xyz, self.user_rpy)
        self.last_action_vel = None

        # 상위 클래스 초기화: 이 호출 이후에 self.IMG_RES 등 필요한 속성이 생성됨
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         episode_len_sec=episode_len_sec,
                         img_res=img_res,
                         img_fps=img_fps)
        # onboard 이미지 저장 경로 생성 (이미 BaseAviary에서 설정됨)
        # 예: self.ONBOARD_IMG_PATH = ...  (이미 BaseAviary.__init__() 내에서 할당)

        # 이제 IMG_RES가 정의되었으므로 dummy_frame을 생성하고 frame_buffer 초기화
        height = int(self.IMG_RES[1])
        width = int(self.IMG_RES[0])
        num_channels = 1 if self.use_gray_scale else 3

        if self.channel_first:  # (C, H, W)
            dummy_frame = np.zeros((num_channels, height, width), dtype=np.uint8)
        else:                   # (H, W, C)
            dummy_frame = np.zeros((height, width, num_channels), dtype=np.uint8)

        # 버퍼를 채워 넣음 (원하는 경우 dummy_frame.copy() 사용)
        self.frame_buffer = [dummy_frame.copy() for _ in range(self.stack_size)]

    def _get_random_drone_pose(self, initial_xyzs=None, initial_rpys=None):
        """
        드론 초기화:
          - 드론 높이는 약 10.0으로 설정
          - 착륙 패드의 위치(self.landing_pad_base_start_pos)를 기준으로
            반경 3.0m 이내의 임의의 오프셋을 주어 LOS 확보
          - 자세: roll, pitch는 ±1° 범위 (조금만 흔들자 일단), yaw는 완전 임의
        """
        # _resetLandingPad()가 먼저 호출되어 pad의 위치가 초기화되어 있다고 가정
        pad_xy = self.landing_pad_base_start_pos[:2]  # 패드의 x, y 좌표

        # 패드 위치를 기준으로 반경 'r'm 이내의 랜덤 오프셋 생성 (균일하게 생성하려면 r <- sqrt(r) 사용; 선형 r은 가운데로 더 몰림)
        if initial_xyzs is None:
            r = np.random.uniform(0, 1.0)
            theta = np.random.uniform(0, 2*np.pi)
            offset_x = r * np.cos(theta)
            offset_y = r * np.sin(theta)
            drone_x = pad_xy[0] + offset_x
            drone_y = pad_xy[1] + offset_y
            drone_z = 4.0  # 고도 10.0 근처
        else:
            drone_x, drone_y, drone_z = initial_xyzs[0]

        initial_xyzs = np.array([[drone_x, drone_y, drone_z]])

        if initial_rpys is None:
            # roll, pitch: ±15° 범위, yaw: [-pi, pi] 범위
            roll = np.deg2rad(np.random.uniform(-1, 1))
            pitch = np.deg2rad(np.random.uniform(-1, 1))
            yaw = np.random.uniform(-np.pi, np.pi)
            initial_rpys = np.array([[roll, pitch, yaw]])

        return initial_xyzs, initial_rpys

    def _resetLandingPad(self):
        """
        착륙 패드 초기화:
          - 패드의 x, y는 (0,0) 근처 임의로 생성 (예: [-1,1] 범위)
          - z는 0.0 고정
          - 패드의 roll, pitch는 0, yaw는 임의 생성
          - 패드의 초기 속도(speed)는 0.5 ~ 2.0 범위에서 임의 생성 (에피소드 동안 일정)
        """
        # 패드의 위치: x, y는 [-0.5, 0.5] 범위, z=0.0
        pad_x = np.random.uniform(-0.5, 0.5)
        pad_y = np.random.uniform(-0.5, 0.5)
        pad_z = 0.0
        self.landing_pad_base_start_pos = np.array([pad_x, pad_y, pad_z])
        self.landing_pad_base_pos = self.landing_pad_base_start_pos.tolist()

        # 패드의 자세: roll, pitch = 0, yaw는 임의
        pad_yaw = np.random.uniform(-np.pi, np.pi)
        self.landing_pad_yaw = pad_yaw  # 이후 업데이트에서 사용
        self.landing_pad_orientation = p.getQuaternionFromEuler([0, 0, pad_yaw])

        # 패드의 속도: 0.0 ~ 1.0 m/s 범위에서 임의 생성
        self.landing_pad_speed = np.random.uniform(0.0, 1.0)

    def _addObstacles(self):
        """
        Overrides _addObstacles() in BaseAviary:,
        움직이는 착륙 패드만 환경에 추가.
        여기서는 착륙 패드를 'parsed_pad.urdf' 파일을 가져옴. 내부에는 base1.obj를 참고하고, 그 내부에는 base1.mtl을 참고하며,
        그 내부에는 입힐 texture를 image 파일을 지정함. 모두 self.assets_path 내에 있어야 함. 복작복작복잡하네.
        Total Joints: 9 (car)
        Link Idx: 0, Name: main_body,              Position: (1.0, 0.0, 0.16)
        Link Idx: 1, Name: rear_bar_link,          Position: (0.84, 0, 0.06)
        Link Idx: 2, Name: back_left_wheel_link,   Position: (0.84, 0.1, 0.06)
        Link Idx: 3, Name: back_right_wheel_link,  Position: (0.84, -0.1, 0.06)
        Link Idx: 4, Name: bar_link,               Position: (1.16, 0, 0.06)
        Link Idx: 5, Name: front_left_wheel_link,  Position: (1.16, -0.1, 0.06)
        Link Idx: 6, Name: front_right_wheel_link, Position: (1.16, 0.1, 0.06)
        Link Idx: 7, Name: holder,                 Position: (1.0, 0.0, 0.235)
        Link Idx: 8, Name: heliport_base,          Position: (1.0, 0.0, 0.2725)

        Helipad: visual shape: (0.675, 0.675, 0), collision shape: (0.5, 0.5, 0)
        """
        pad_urdf = os.path.join(self.assets_path, "parsed_pad.urdf")
        # print("PyBullet is searching in:", os.getcwd())  # Check current directory
        self.pad_link_center_idx = 8
        self.landing_pad_id = p.loadURDF(fileName=pad_urdf,
                                         basePosition=self.landing_pad_base_pos,
                                         baseOrientation=self.landing_pad_orientation,
                                         physicsClientId=self.CLIENT)

    def _updateLandingPad(self):
        """
        Updates pad position every step.:
          - 패드가 초기화된 yaw 방향(heading)으로 일정 속도(self.landing_pad_speed)로 직진.
        """
        dt = self.TIMESTEP  # 한 스텝의 시간 간격
        dx = self.landing_pad_speed * dt * np.cos(self.landing_pad_yaw)
        dy = self.landing_pad_speed * dt * np.sin(self.landing_pad_yaw)
        self.landing_pad_base_pos[0] += dx
        self.landing_pad_base_pos[1] += dy
        # z는 0.0으로 고정
        p.resetBasePositionAndOrientation(self.landing_pad_id,
                                          self.landing_pad_base_pos,
                                          self.landing_pad_orientation,
                                          physicsClientId=self.CLIENT)

    def _get_pad_center_position(self):
        # linkWorldPosition: (vec3, list of 3 floats): Cartesian position of center of mass
        position = p.getLinkState(self.landing_pad_id, self.pad_link_center_idx, physicsClientId=self.CLIENT)[0]
        return np.array(position, dtype=np.float64)  # (3,)

    def _get_pad_center_orientation(self, quaternion=True):
        orientation_in_quaternion = p.getLinkState(self.landing_pad_id, self.pad_link_center_idx, physicsClientId=self.CLIENT)[1]
        if quaternion:
            # linkWorldOrientation: (vec4, list of 4 floats): Cartesian orientation of center of mass in XYZW quaternion
            return np.array(orientation_in_quaternion, dtype=np.float64)  # (4,)
        else:
            # convert quaternion to euler angles (roll, pitch, yaw) -- ROS URDF convention
            return np.array(p.getEulerFromQuaternion(orientation_in_quaternion), dtype=np.float64)  # (3,)

    def reset(self):
        """
        환경 리셋:
          - 드론 초기 위치와 자세를 _reset_drone()을 통해 랜덤으로 재설정
          - 착륙 패드 초기화 및 위치 업데이트
          - 상위 환경(super) reset 호출
          - 카메라 이미지(초기 프레임)를 받아서 프레임 버퍼를 stack_size만큼 채움
          - stacked observation 반환
        """
        self.last_action_vel = np.zeros(3)  # 마지막으로 적용된 속도 명령

        # 드론 초기 위치/자세를 랜덤으로 재설정
        initial_xyzs, initial_rpys = self._get_random_drone_pose(self.user_xyz, self.user_rpy)
        self.INIT_XYZS = initial_xyzs
        self.INIT_RPYS = initial_rpys

        # 착륙 패드 리셋 및 초기 업데이트
        self._resetLandingPad()
        self._updateLandingPad()

        # 상위 환경(super)의 reset() 호출 (여기서 p.resetSimulation() 등이 실행됨)
        obs = super().reset()

        return obs

    def _getDroneImages(self, nth_drone, segmentation: bool=True):
        if self.IMG_RES is None:
            print("[ERROR] in VisionLandingAviary._getDroneImages(), remember to set self.IMG_RES to np.array([width, height])")
            exit()

        # 1) 드론 오리엔테이션(쿼터니언) → 회전행렬
        quat_drone = self.quat[nth_drone, :]
        rot_mat = np.array(p.getMatrixFromQuaternion(quat_drone)).reshape((3,3))

        # 2) 드론에서의 카메라 offset, 보고싶은 방향, up 벡터
        camera_offset = np.array([0., 0., 0.02])        # 드론 중심 대비 카메라 위치
        camera_target_offset = np.array([0., 0., -1.]) # 아래 방향을 보고 싶다면 -z
        camera_up_in_drone_frame = np.array([0., 1., 0.])

        # 3) 월드좌표계로 변환
        cameraEye = self.pos[nth_drone] + rot_mat.dot(camera_offset)
        target    = self.pos[nth_drone] + rot_mat.dot(camera_target_offset)
        upVector  = rot_mat.dot(camera_up_in_drone_frame)

        DRONE_CAM_VIEW = p.computeViewMatrix(
            cameraEyePosition=cameraEye,
            cameraTargetPosition=target,
            cameraUpVector=upVector,
            physicsClientId=self.CLIENT
        )

        DRONE_CAM_PRO = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=1.0,
            nearVal=0.03,
            farVal=200.0
        )

        SEG_FLAG = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if segmentation else p.ER_NO_SEGMENTATION_MASK

        [w, h, rgb, dep, seg] = p.getCameraImage(
            width=self.IMG_RES[0],
            height=self.IMG_RES[1],
            shadow=1,
            viewMatrix=DRONE_CAM_VIEW,
            projectionMatrix=DRONE_CAM_PRO,
            flags=SEG_FLAG,
            physicsClientId=self.CLIENT
        )
        ...
        return rgb, dep, seg

    def _get_stacked_obs(self, rgb):
        """
        프레임 버퍼에 저장된 최신 이미지들을 채널 방향으로 이어붙여 stacked observation 생성.
        예: 각 프레임이 (H, W, 3)이라면 stacked_obs의 shape는 (H, W, 3*stack_size)가 됨.
        """
        # RGB 채널만 사용
        if self.channel_first:
            # 채널을 마지막 축에서 첫 번째 축으로 이동: (H, W, 3) → (3, H, W)
            frame = np.moveaxis(rgb[:, :, :3], -1, 0)
        else:
            frame = rgb[:, :, :3]

        # 그레이스케일 변환 적용 (channel_first 여부를 인자로 전달)
        if self.use_gray_scale:
            frame = rgb2gray(frame, norm=False, keepdims=True, channel_first=self.channel_first)

        # 프레임 버퍼 업데이트
        self.frame_buffer.pop(0)
        self.frame_buffer.append(frame)

        # 채널 방향으로 이어붙이기
        channel_axis = 0 if self.channel_first else -1
        return np.concatenate(self.frame_buffer, axis=channel_axis)

    def _computeObs(self):
        """
        매 스텝마다 호출되어 observation을 생성함.
          - 드론 카메라 이미지(RGB)를 받아 (알파 채널 제외) 새로운 프레임 생성
          - 프레임 버퍼를 업데이트(가장 오래된 프레임 제거 후 새 프레임 추가)
          - record=True인 경우, PNG 파일로 onboard 이미지를 저장함
          - stack된 이미지를 반환
        """
        rgb, _, _ = self._getDroneImages(0, segmentation=False)
        # rgb, _, _ = self._getDroneImagesWithOrientation(0, segmentation=False)

        # onboard 이미지 저장 (record=True이면)
        if self.RECORD and (self.step_counter % self.IMG_CAPTURE_FREQ == 0):
            self._exportImage(img_type=ImageType.BW if self.use_gray_scale else ImageType.RGB,
                              img_input=rgb,
                              path=self.ONBOARD_IMG_PATH,
                              frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ))

        # 드론 상태 정보 (linear vel, angular vel, orientation)를 추가하여 observation 반환
        state = np.hstack([self.quat[0, :], self.vel[0, :], self.last_action_vel])
        assert state.shape == (10,), f"Invalid drone_state shape in _computeObs: {state.shape}"

        return {"images": self._get_stacked_obs(rgb), "drone_state": state}
        # return self._get_stacked_obs(rgb)

    def _observationSpace(self):
        """
        Observation space 재정의:
          - 각 카메라 프레임의 크기는 IMG_RES (예: [84, 84])
          - RGB 이미지이면 한 프레임당 채널 수는 3, 그레이스케일이면 1
          - stack_size 프레임을 쌓으므로 최종 shape는:
                * channel_first=True → (channels*stack_size, height, width)
                * channel_first=False → (height, width, channels*stack_size)
        """
        height = int(self.IMG_RES[1])
        width = int(self.IMG_RES[0])
        channels = 1 if self.use_gray_scale else 3

        if self.channel_first:
            shape = (channels * self.stack_size, height, width)
        else:
            shape = (height, width, channels * self.stack_size)

        return spaces.Dict({"images": spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8),
                            "drone_state": spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)})
        # return spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

    def _actionSpace(self):
        """Returns the action space of the environment.
        Returns
        ndarray
            A Box() of size 1, 3, 4, or 6 depending on the action type.
        """
        if self.ACT_TYPE == ActionType.VEL:
            size = 3
        elif self.ACT_TYPE == ActionType.RPM:
            size = 4
        elif self.ACT_TYPE in [ActionType.TUN, ActionType.DYN, ActionType.PID, ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
            raise NotImplementedError("Action type not implemented yet.")
        else:
            raise ValueError(f"Invalid ActionType: {self.ACT_TYPE} in VisionLandingAviary._actionSpace()")

        return spaces.Box(low=-1*np.ones(size, dtype=np.float32),
                          high=np.ones(size, dtype=np.float32),
                          dtype=np.float32)
        # return spaces.Box(
        #     low=-1 * np.ones(size, dtype=np.float64),
        #     high=np.ones(size, dtype=np.float64),
        #     dtype=np.float64
        # )

    def _preprocessAction(self, action):
        if self.ACT_TYPE == ActionType.VEL:
            # action: shape=(3,) in [-1, 1]
            # scale it to the desired velocity in m/s
            scaled_action = np.clip(action, -1, 1)  # 혹시 모를 안전장치
            self.last_action_vel = scaled_action
            target_vel_input = self.SPEED_LIMIT * scaled_action
            curr_vel = self.vel[0, :]

            # # Blend the target velocity input with the current velocity
            # # This is to prevent the drone from stopping abruptly; but not converging sometimes...
            target_vel = 0.15 * target_vel_input + 0.85 * curr_vel

            state = self._getDroneStateVector(0)
            rpm, _, _ = self.ctrl.computeControl(
                control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=state[0:3],  # 위치는 고정 (현재 위치)
                target_rpy=np.array([0,0,state[9]]),  # 현재 yaw 유지
                target_vel=target_vel
            )
            return rpm
        elif self.ACT_TYPE == ActionType.RPM:
            return np.array(self.HOVER_RPM * (1+0.05*action))
        elif self.ACT_TYPE in [ActionType.TUN, ActionType.PID, ActionType.DYN, ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
            raise NotImplementedError("Action type not implemented yet.")
        else:
            raise ValueError(f"Invalid ActionType: {self.ACT_TYPE}")

    def _curriculum_v1_rewards(self):
        """
        Curriculum v1 - difficulty 1–4 as follows:
          - Difficulty 1: Visibility reward only
          - Difficulty 2: visibility + landing/crashing (no crash penalty i.e., no negative reward)
          - Difficulty 3: visibility와 landing/crashing (crash penalty applied)
          - Difficulty 4: vanilla reward (use all reward functions)
        """
        if self.difficulty == 1:
            # visibility only
            return self._compute_reward_visibility()

        elif self.difficulty == 2:
            # visibility + landing/crashing (no negative reward; i.e. no crash penalty)
            r_vis = self._compute_reward_visibility()
            if r_vis < 0:
                return r_vis  # 시야가 확보되지 않은 경우는 그대로 음수 반환
            # landing/crashing 에서 음수(패널티)는 0으로 처리 (no_crash_penalty=True)
            r_land = self._compute_reward_landing_or_crashing(no_crash_penalty=True)
            return r_vis + r_land

        elif self.difficulty == 3:
            # visibility + landing/crashing (음수 보상 적용)
            r_vis = self._compute_reward_visibility()
            if r_vis < 0:
                return r_vis
            r_land = self._compute_reward_landing_or_crashing(no_crash_penalty=False)
            if r_land < 0:
                return r_land
            return r_vis + r_land

        elif self.difficulty == 4:
            # vanilla reward 함수: visibility, landing/crashing, vertical velocity, horizontal distance 모두 사용
            return self._vanilla_reward_function()

        else:
            raise ValueError(f"Invalid curriculum difficulty: {self.difficulty}")

    def _vanilla_reward_function(self):
        # 1. Check visibility
        reward_visibility = self._compute_reward_visibility()
        if reward_visibility < 0:
            return reward_visibility

        # 2. Check Landing/Crashing
        reward_landing_or_crashing = self._compute_reward_landing_or_crashing()
        if reward_landing_or_crashing < 0:
            return reward_landing_or_crashing

        # 3. Compute vertical velocity reward (safety)
        reward_vertical_velocity = self._compute_vertical_velocity_reward()
        # 4. Compute horizontal distance reward
        reward_horizontal_dist = self._compute_horizontal_dist_reward()

        # 5. Combine rewards
        reward = 0.6 * reward_horizontal_dist + 1.0 * reward_vertical_velocity
        reward += reward_visibility + reward_landing_or_crashing
        return reward

    def set_difficulty(self, new_diff: int):
        """This setter is provided to change the difficulty during the training."""
        self.difficulty = new_diff

    def _computeReward(self):
        return self._curriculum_v1_rewards()

    def _compute_horizontal_dist_reward(self):
        # Get relative xy distance b/w the drone and the helipad
        rel_xy_dist = np.linalg.norm(self.pos[0, 0:2] - self._get_pad_center_position()[0:2])

        # Compute reward
        rel_xy_thresh = 10.0
        rho = 30.0
        normalized_rel_xy_dist = (rel_xy_thresh - rel_xy_dist) / rel_xy_thresh
        if normalized_rel_xy_dist > 0:  # within the threshold
            return (rho ** normalized_rel_xy_dist - 1) / (rho - 1)  # in [0, 1]
        else:  # beyond the threshold
            return 0.0

    def _compute_vertical_velocity_reward(self):
        alpha = 30.0
        desired_z_vel = -0.5
        assert abs(desired_z_vel) < self.SPEED_LIMIT[2], "Desired z velocity should be within the speed limit."

        drone_z_vel = self.vel[0, 2]  # z velocity

        # (1) Move too fast
        if abs(drone_z_vel) / self.SPEED_LIMIT[2] > 1.1:
            return 0.0
        # (2) Ascending, which isn't desired
        if drone_z_vel > 0:
            return -0.1
        # (3) Safe descending: max reward at the desired z-velocity
        if desired_z_vel < drone_z_vel <= 0:
            return (alpha ** (drone_z_vel / desired_z_vel) - 1) / (alpha - 1)
        # (4) Descending faster than desired
        elif drone_z_vel <= desired_z_vel:
            return -0.01
        else:
            raise "VisionLandingAviary env._compute_vertical_velocity_reward(): This should not happen!"

    def _compute_reward_landing_or_crashing(self, no_crash_penalty=False):
        drone_id = self.DRONE_IDS[0]
        drone_altitude = self.pos[0, 2]

        if drone_altitude >= self.pad_height and p.getContactPoints(bodyA=drone_id, physicsClientId=self.CLIENT) != ():
            print(' @@  VisionLandingAviary env: Landed!')
            return 180.0
        elif drone_altitude < self.pad_height and p.getContactPoints(bodyA=drone_id, physicsClientId=self.CLIENT) != ():
            print(' @@  VisionLandingAviary env: Crashed!')
            if no_crash_penalty:
                return 0.0
            else:
                return -1.0
        else:  # Hasn't landed or crashed yet
            return 0.0

    def _compute_reward_visibility(self):
        """Returns non-negative reward if LOS; otherwise, negative reward."""
        pad_position = self._get_pad_center_position()  # numpy (3,)
        # if self._check_los(pad_position):
        # if self._check_los_segmentation():
        if self._check_los_camera_polygon_vs_pad_box():
            return 0.0
        else:
            # print("  !!  [VisionLandingAviary] env: Out of LOS!")
            return -0.01

    def _check_los_camera_polygon_vs_pad_box(self):
        """
        카메라 중앙 벡터 & 코너 벡터를 패드 평면으로 연장해 얻은 다각형과
        패드의 충돌 박스(0.5×0.5) 사각형이 2D 평면에서 교차하는지 확인.
        교차(또는 포함)하면 LOS가 있다고 봄.
        """
        # [1] 드론(카메라) 위치·방향 구하기
        drone_pos = self.pos[0]           # shape = (3,)
        drone_quat = self.quat[0]         # shape = (4,) (x, y, z, w)라 가정
        rot_world_to_drone = Rotation.from_quat(drone_quat).inv()

        # [2] 패드 평면의 중심점, 쿼터니언, 법선(normal) 등 구하기
        pad_center_3d = self._get_pad_center_position()                # (3,)
        pad_quat = self._get_pad_center_orientation(quaternion=True)   # (x, y, z, w)
        R_pad = np.array(p.getMatrixFromQuaternion(pad_quat)).reshape(3, 3)
        # 패드 평면의 법선(로컬 z축을 회전한 벡터라고 가정)
        #  - pad 로컬에서 z축은 (0,0,1)이라 가정 → 실제 urdf 상 어떻게 정의됐는지 확인 필요
        pad_normal = R_pad @ np.array([0, 0, 1])   # shape=(3,)

        # [3] 카메라 코너(및 중앙) 방향 정의 (드론 바디 기준)
        # Ex) 'center': (0, 0, -1),
        #     '4 corners': (±1, ±1, -1) 형태 등을 사용. (i.e. fov=90)
        # self.fov: vertical FOV (degrees)
        # self.IMG_RES: np.array([w, h])
        fov_margin_deg = 2.0  # 카메라 이미지보다 조금 더 안쪽으로 들어오게 계산
        fov_rad = np.radians(self.fov - fov_margin_deg)
        aspect_ratio = self.IMG_RES[0] / self.IMG_RES[1]
        v = np.tan(fov_rad / 2.0)  # 수직 한계 (top/bottom)
        h = aspect_ratio * v       # 수평 한계 (left/right)
        #
        dirs_body = [
            np.array([0., 0., -1.]),  # center
            np.array([ h,  v, -1.]),  # top-right
            np.array([-h,  v, -1.]),  # top-left
            np.array([-h, -v, -1.]),  # bottom-left
            np.array([ h, -v, -1.]),  # bottom-right
        ]

        # Normalization
        dirs_body = [d / np.linalg.norm(d) for d in dirs_body]

        # [4] 각 벡터를 월드로 변환 + 평면 교차점 찾기
        intersection_points = []
        for d_body in dirs_body:
            d_world = rot_world_to_drone.inv().apply(d_body)  # (3,)
            pt = line_plane_intersection(
                line_point=drone_pos,
                line_dir=d_world,
                plane_point=pad_center_3d,
                plane_normal=pad_normal
            )
            intersection_points.append(pt)

        # 교차점 0번은 "중앙" 교차점 A, 나머지 4개가 C, D, E, F에 해당
        A = intersection_points[0]  # 중심
        corners_3d = intersection_points[1:]  # 4개 모서리

        # [5] 먼저 "중심점 A"가 패드 0.5×0.5 박스 안에 있으면 LOS=True
        #   -> 패드 평면 2D 좌표로 투영
        if A is not None:
            A_2d = project_point_on_pad_plane(A, pad_center_3d, R_pad)  # 아래에서 구현
            if inside_pad_box(A_2d, half_size=0.25):
                return True

        # [6] corners_3d 중 None 있으면(카메라 뒤?), 유효한 점만 사용
        valid_corners_3d = [c for c in corners_3d if c is not None]
        if len(valid_corners_3d) == 0:
            # 카메라 frustum과 패드 평면이 교차X → 볼 수 없음
            return False

        # [7] 유효한 코너를 패드 평면 2D로 투영 (x,y 좌표)
        corners_2d = [project_point_on_pad_plane(c, pad_center_3d, R_pad)
                      for c in valid_corners_3d]

        # "카메라 폴리곤"이라 할 수 있는 2D 점들 corners_2d
        # → 보통 (C->D->E->F) 순으로 이어야 하지만
        #   정렬되지 않았을 수 있으니, 볼록다각형 정렬이 필요함
        #   (convex hull을 취하거나, 시계/반시계로 소팅 등)
        cam_poly_2d = order_points_convex_polygon(corners_2d)

        # [8] 패드 충돌박스도 2D 사각형 (±0.25, ±0.25)
        pad_box_2d = np.array([
            [ 0.25,  0.25],
            [-0.25,  0.25],
            [-0.25, -0.25],
            [ 0.25, -0.25],
        ])

        # [9] 다각형 교차 여부 검사
        if polygons_intersect_2d(cam_poly_2d, pad_box_2d):
            return True

        return False

    def _computeDone(self):
        """
        종료 조건:
        - 드론이 다른 것과 충돌하면 done
        - 에피소드 시간 초과 (self.EPISODE_LEN_SEC)
        """
        # 드론이 다른것과 충돌하면 done
        # Note: 땅과 충돌해도 끝남;;
        if p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ():
            return True
        # 에피소드 시간 초과
        # Note: self.step_counter hasn't been updated in step() yet;
        #       So, it is smaller than actual step count by self.AGGR_PHY_STEPS at this line.
        if (self.step_counter + self.AGGR_PHY_STEPS) >= self.EPISODE_LEN_SEC * self.SIM_FREQ:
            return True

        return False

    def _computeInfo(self):
        info = {}
        return info

    def step(self, action):
        """
        한 스텝 진행 시,
          1. 먼저 착륙 패드의 위치를 업데이트한 후,
          2. 상위 환경(super)의 step()을 호출하여 시뮬레이션을 진행하고,
          3. observation, reward, done, info를 반환함.
        """
        self._updateLandingPad()
        return super().step(action)

    def _computeReward_backup_thanks_to_Pawel(self):
        # This is just a backup; not used in the current implementation; ignore this method
        # Parameters
        desired_z_vel = -0.5
        alpha = 30.0
        xy_must_smaller_than = 10.0
        rho = 30.0

        # Get drone and UGV positions and velocities
        UGV_pos = self._get_pad_center_position()  # p.getLinkState(self.landing_pad_id,..) in np.array
        drone_state = self._getDroneStateVector(0)  # nth_drone=0
        drone_position = drone_state[0:3]  # x, y, z
        drone_velocity = drone_state[10:13]  # vx, vy, vz (linear velocity)

        # Get distance errors: xy, z, and angle
        distance_xy = np.linalg.norm(drone_position[0:2]-UGV_pos[0:2])
        distance_z = np.linalg.norm(drone_position[2:3]-UGV_pos[2:3])
        angle = np.rad2deg(np.arctan2(distance_xy,distance_z))  # be careful about the angle range

        # Check if drone follows the desired z velocity
        # moves_down_and_safe_in_z==True if (1) moves down and (2) slower than the desired speed (i.e. |desired_z_vel|)
        moves_down_and_safe_in_z = (0 >= drone_velocity[2]) * (drone_velocity[2] > desired_z_vel)

        # (1) Compute reward: Vertical velocity (safety)
        if moves_down_and_safe_in_z:
            reward_z_vel = (alpha**(drone_velocity[2]/desired_z_vel) -1)/(alpha -1)
        else:  # Penalize if drone moves up or too fast
            if abs(drone_velocity[2])/self.SPEED_LIMIT[2] > 1.1:
                reward_z_vel = 0
            else:
                if drone_velocity[2] <= desired_z_vel:
                    reward_z_vel = -0.01
                else:
                    reward_z_vel = -0.1

        # (2) Compute reward: Horizontal distance
        if distance_xy < xy_must_smaller_than:
            normalized_distance_xy = (xy_must_smaller_than - distance_xy) / (xy_must_smaller_than)
            reward_xy = (rho**normalized_distance_xy -1)/(rho -1)
        else:  # Too far!
            reward_xy = 0 #-distance_xy

        # (3) Get total reward
        combined_reward = 0.6 * reward_xy + 1.0 * reward_z_vel

        # (4) Landing/Crashing
        drone_id = self.DRONE_IDS[0]
        if drone_position[2] >= 0.275 and p.getContactPoints(bodyA=drone_id, physicsClientId=self.CLIENT) != ():
            print('landed!')
            combined_reward =  140 + combined_reward
        elif drone_position[2]  < 0.275 and p.getContactPoints(bodyA=drone_id, physicsClientId=self.CLIENT) != ():
            print('crashed!')
            combined_reward = -1
        else:
            combined_reward =  combined_reward

        # (5) Visibility
        distance_x = np.abs(drone_position[0]-UGV_pos[0])
        distance_y = np.abs(drone_position[1]-UGV_pos[1])
        if np.abs(angle) > 30 and (distance_y > 0.8 and distance_x > 0.8):
            combined_reward = -0.01

        return combined_reward

    def _getDroneImages_org(self, nth_drone, segmentation: bool=True):
        if self.IMG_RES is None:
            print("[ERROR] in VisionLandingAviary._getDroneImages(), remember to set self.IMG_RES to np.array([width, height])")
            exit()

        # 기존의 드론 회전 행렬 관련 코드는 제거하거나 무시
        # rot_mat = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)

        # 카메라 위치: 드론 중심에서 필요에 따라 약간 아래로 배치 (예: [0, 0, 0.0] 또는 [-0.1] 등)
        # Do figure out the best position for the camera for your application
        cameraEye = self.pos[nth_drone, :] + np.array([0, 0, 0.087])

        # 카메라 목표: 드론의 위치에서 충분히 아래쪽(예: 1000m 아래)로 설정
        target = self.pos[nth_drone, :] + np.array([0, 0, -1000])

        # 업 벡터: 카메라 이미지의 “위쪽” 방향을 결정 (여기서는 [0, 1, 0] 사용)
        upVector = [0, 1, 0]

        DRONE_CAM_VIEW = p.computeViewMatrix(
            cameraEyePosition=cameraEye,
            cameraTargetPosition=target,
            cameraUpVector=upVector,
            physicsClientId=self.CLIENT
        )

        DRONE_CAM_PRO = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=1.0,
            nearVal=0.1,    # near plane 값을 적절히 조절 (예: 드론 크기 고려)
            farVal=1000.0
        )

        SEG_FLAG = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if segmentation else p.ER_NO_SEGMENTATION_MASK
        [w, h, rgb, dep, seg] = p.getCameraImage(
            width=self.IMG_RES[0],
            height=self.IMG_RES[1],
            shadow=1,
            viewMatrix=DRONE_CAM_VIEW,
            projectionMatrix=DRONE_CAM_PRO,
            flags=SEG_FLAG,
            physicsClientId=self.CLIENT
        )
        rgb = np.reshape(rgb, (h, w, 4))
        dep = np.reshape(dep, (h, w))
        seg = np.reshape(seg, (h, w))
        return rgb, dep, seg

    def convert_external_images_to_video(self, output_file="external_video.mp4", fps=30):
        convert_images_to_video(self.IMG_PATH, output_file, fps=fps, pattern="frame_%d.png")

    def convert_onboard_images_to_video(self, output_file="onboard_video.mp4", fps=30):
        convert_images_to_video(self.ONBOARD_IMG_PATH, output_file, fps=fps, pattern="frame_%d.png")

    # About to deprecated
    def _check_los(self, pad_position):
        """
        Returns True if the drone has line-of-sight to the pad; otherwise, False.
        """
        drone_pos = self.pos[0, :]          # shape: (3,)
        drone_quat = self.quat[0, :]        # shape: (4,) 가정: (x, y, z, w) 또는 (w, x, y, z)

        # 월드 좌표계에서 타겟 벡터
        target_vec_world = pad_position - drone_pos  # shape: (3,)

        # 쿼터니언 → 회전행렬(또는 Rotation 객체)
        # Scipy는 기본적으로 [x, y, z, w] 순서를 받음. (만약 [w, x, y, z]라면 순서 맞춰야 함)
        rot_world_to_drone = Rotation.from_quat(drone_quat)

        # 월드 → 드론 바디로 벡터 변환
        target_vec_drone = rot_world_to_drone.inv().apply(target_vec_world)

        # 드론 바디에서 카메라가 -Z 방향을 본다고 가정하므로,
        # z가 음수이면 카메라가 바라보는 '앞쪽(아래쪽)'에 위치하게 됨
        x_d = target_vec_drone[0]
        y_d = target_vec_drone[1]
        z_d = target_vec_drone[2]

        # 카메라가 -Z쪽을 본다고 할 때, z_d가 양수라면 카메라의 "뒷면"에 있는 것
        if z_d > 0:
            return False

        # FOV 체크
        # 수평/수직 시야각이 self.fov로 동일하다고 할 때,
        # x, y 각 축에 대해 시야각을 초과하는지 확인하면 됨.
        # 각도 계산은 arctan2(수평방향, 종방향) 사용
        # z축이 음수이므로 -z_d를 분모로 사용 (z_d가 음수이므로 -z_d는 양수)
        half_fov = self.fov / 2.0

        # arctan2의 결과에 abs()를 취해서 카메라 중앙축으로부터 떨어진 각도를 구함
        angle_x = np.degrees(np.arctan2(abs(x_d), -z_d))  # 드론 바디 기준
        angle_y = np.degrees(np.arctan2(abs(y_d), -z_d))

        # x, y 방향 모두 fov/2 이내면 카메라 프레임 안에 있는 것
        if (angle_x <= half_fov) and (angle_y <= half_fov):
            return True
        else:
            return False

    # About to deprecated
    def _check_los_segmentation(self):
        """
        세그멘테이션을 활용하여 착륙 패드(landing_pad_id)의 link가
        카메라 프레임 안에 하나라도 찍혀 있으면 LOS=True 반환
        """
        # NOT TESTED YET
        # 세그멘테이션 사용해서 이미지 받아오기
        # segmentation=True 로 호출해야 ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX 모드로 동작합니다.
        _, _, seg = self._getDroneImages(nth_drone=0, segmentation=True)

        # seg 배열의 각 픽셀에는 objectUniqueId와 linkIndex가 비트로 encoding되어 있음
        # 공식 문서에 따르면, segPixelValue = objectUniqueId << 24 + linkIndex << 16 + ...
        # 여기서 objectUniqueId를 얻으려면 (segPixelValue & ((1 << 24) - 1)) >> 16 이런 식으로 bit마스크를 해주어야 합니다.

        # 하지만 착륙 패드의 link들도 구별하려면, linkIndex까지 확인해야 합니다.
        # pad_id = self.landing_pad_id
        pad_uid = self.landing_pad_id  # 착륙 패드의 유니크 ID

        height, width = seg.shape[:2]
        for y in range(height):
            for x in range(width):
                pix = seg[y, x]
                # 오브젝트 ID
                object_uid = (pix & ((1 << 24) - 1)) >> 16
                if object_uid == pad_uid:
                    # 착륙 패드의 어느 link든 한 픽셀이라도 카메라에 찍혔다면 LOS = True
                    return True

        return False


class VisionLandingAviaryLCfirst(VisionLandingAviary):
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 freq: int = 300,
                 aggregate_phy_steps: int = 10,
                 gui: bool = False,
                 record: bool = False,
                 obs: ObservationType = ObservationType.BW,
                 act: ActionType = ActionType.VEL,
                 channel_first: bool = True,
                 stack_size: int = 4,
                 fov: float = 80.0,
                 img_res: np.ndarray = np.array([84, 84]),
                 img_fps: int = 30,
                 episode_len_sec: float = 30.0,
                 include_drone_state: bool = True,
                 difficulty: int = 4,
                 curriculum_configs: dict = None,
                 ):
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         channel_first=channel_first,
                         stack_size=stack_size,
                         fov=fov,
                         img_res=img_res,
                         img_fps=img_fps,
                         episode_len_sec=episode_len_sec,
                         include_drone_state=include_drone_state,
                         difficulty=difficulty,
                         curriculum_configs=curriculum_configs)

    def _vanilla_reward_function(self):
        # 2. Check Landing/Crashing
        reward_landing_or_crashing = self._compute_reward_landing_or_crashing()
        if reward_landing_or_crashing < 0:
            return reward_landing_or_crashing

        # 1. Check visibility
        reward_visibility = self._compute_reward_visibility()
        if reward_visibility < 0:
            return reward_visibility

        # 3. Compute vertical velocity reward (safety)
        reward_vertical_velocity = self._compute_vertical_velocity_reward()
        # 4. Compute horizontal distance reward
        reward_horizontal_dist = self._compute_horizontal_dist_reward()

        # 5. Combine rewards
        reward = 0.6 * reward_horizontal_dist + 1.0 * reward_vertical_velocity
        reward += reward_visibility + reward_landing_or_crashing
        return reward

