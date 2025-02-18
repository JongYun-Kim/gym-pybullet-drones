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
      - 평면에 착륙하면 z = 0.135 (base==self.pos)
      - 패드 착륙
        - 이론상: z = 0.2860 == 0.2725 + 0.0135 (heliport_base + drone_base_height)
        - 실험서: z = 0.28754655
    - 헬리패드: 움직이는 착륙 패드 (simple car + helipad texture from parsed_pad.urdf in assets dir)
    - Body ID: 0, Name: plane  as idk yet
    - Body ID: 1, Name: cf2    as self.DRONE_IDS
    - Body ID: 2, Name: car    as self.landing_pad_id
    - Observation: 드론 카메라에서 촬영한 RGB 이미지(알파 채널 제외)를 4 프레임 stack
    - Action: ActionType.RPM or VEL
    - Reward: Visibility + Safety (vertical vel) + Landing/Crashing
    - 기록: 환경 외부 카메라 기록(BaseAviary의 record 옵션) 외에도, onboard 카메라 이미지가 PNG로 저장되며, 후에 ffmpeg로 동영상 변환 가능.
"""
# TODOs:
# - [o] rgb to grey scale
# - [o] Check observation stack interval
#   - 24 Hz 로 이미지 촬영; 4 프레임 크기의 buffer를 새로운 프레임으로 업데이트
#   - aggregate_phy_steps=10 을 하면 freq=240Hz에서 알아서 매 observation 마다 RL-action 을 할 기회를 줌 (빠른 simulation)
# - [o] Replace the reward function with Pawel's
# - [o] Check done condition
# - [ ] Check the curricula
# - [ ] Implement the curriculum learning
# - [ ] Randomize landing pad direction (yaw)
# - [ ] Randomize drone initial position and orientation
#   - [ ] LOS check (vision 범위에서 시작해야함)
# - [ ] Laters:
#   - [ ] Make a configuration for the desired z velocity: @ the def of 'self.SPEED_LIMIT' in BaseSingleAgentAviary
#   - [ ] Randomize landing pad texture
#   - [>] Check getDroneImage method (rotation wise...)
# (2) _getDroneImages 메서드 제대로 된건지 확인 하기 (fov 등은 잘 되는데)
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


class VisionLandingAviary(BaseSingleAgentAviary):
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 freq: int = 240,
                 aggregate_phy_steps: int = 10,
                 gui: bool = False,
                 record: bool = False,
                 obs: ObservationType = ObservationType.BW,
                 act: ActionType = ActionType.VEL,
                 channel_first: bool = True,     # nn.Conv2d() 사용시 channel_first=True
                 stack_size: int = 4,            # 이미지 프레임 stack 개수
                 fov: float = 60.0,              # drone 카메라 시야각 (degree)
                 img_res: np.ndarray = np.array([84, 84]),  # original: np.array([64, 48])
                 img_fps: int = 24,
                 episode_len_sec: float = 5.0,   # 에피소드 길이 (초)
                 include_drone_state: bool = True,
                 ):
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
        initial_xyzs, initial_rpys = self._reset_drone(initial_xyzs, initial_rpys)

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

    def _reset_drone(self, initial_xyzs=None, initial_rpys=None):
        # TODO: randomize initial_xyzs and initial_rpys
        # TODO: use landing pad position to give los in the initial observation
        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 1.0]])

        if initial_rpys is None:
            initial_rpys = np.array([[0.0, 0.0, 0.0]])

        return initial_xyzs, initial_rpys

    def _resetLandingPad(self):
        """
        착륙 패드 관련 파라미터 초기화:
            - 초기 위치: [x, y, z] (높이는 약간 올려서 충돌 판정을 피할 수도 있음)
            - 이동 반경 및 속도: 원형 궤적으로 움직이도록 설정
        """
        self.landing_pad_base_start_pos = np.array([0.0, 0.0, 0.0])
        self.landing_pad_amplitude = 1.0   # 원의 반지름 (미터)
        self.landing_pad_omega = 0.2       # 각속도 (rad/s)
        self.landing_pad_base_pos = self.landing_pad_base_start_pos.tolist()

    def _addObstacles(self):
        """
        BaseAviary의 _addObstacles()를 오버라이드하여,
        움직이는 착륙 패드만 환경에 추가한다.
        여기서는 착륙 패드를 'parsed_pad.urdf' 파일을 가져옴. 내부에는 base1.obj를 참고하고, 그 내부에는 base1.mtl을 참고하며,
        그 내부에는 입힐 texture를 image 파일을 지정함. 모두 self.assets_path 내에 있어야 함. 복작복작복잡하네.
        Total Joints: 9 (car)
        Link Index: 0, Name: main_body,              Position: (1.0, 0.0, 0.16)
        Link Index: 1, Name: rear_bar_link,          Position: (0.84, 0, 0.06)
        Link Index: 2, Name: back_left_wheel_link,   Position: (0.84, 0.1, 0.06)
        Link Index: 3, Name: back_right_wheel_link,  Position: (0.84, -0.1, 0.06)
        Link Index: 4, Name: bar_link,               Position: (1.16, 0, 0.06)
        Link Index: 5, Name: front_left_wheel_link,  Position: (1.16, -0.1, 0.06)
        Link Index: 6, Name: front_right_wheel_link, Position: (1.16, 0.1, 0.06)
        Link Index: 7, Name: holder,                 Position: (1.0, 0.0, 0.235)
        Link Index: 8, Name: heliport_base,          Position: (1.0, 0.0, 0.2725)

        Helipad: visual shape: (0.675, 0.675, 0), collision shape: (0.5, 0.5, 0)
        """
        pad_urdf = self.assets_path + "/parsed_pad.urdf"
        yaw = np.random.uniform(-np.pi/12.0, np.pi/12.0)
        pad_start_orientation_euler = [0,0, yaw]
        pad_start_orientation_quaternion = p.getQuaternionFromEuler(pad_start_orientation_euler)

        # print("PyBullet is searching in:", os.getcwd())  # Check current directory
        self.pad_link_center_idx = 8
        self.landing_pad_id = p.loadURDF(fileName=pad_urdf,
                                         basePosition=self.landing_pad_base_pos,
                                         baseOrientation=pad_start_orientation_quaternion,
                                         physicsClientId=self.CLIENT)

    def _updateLandingPad(self):
        """
        매 스텝마다 착륙 패드의 위치를 원형 궤적으로 업데이트함.
        (시간 t에 따라 x = x0 + A*cos(omega*t), y = y0 + A*sin(omega*t))
        그리고 p.resetBasePositionAndOrientation()를 통해 패드의 위치를 갱신함.
        """
        t = self.step_counter * self.TIMESTEP
        # x = self.landing_pad_base_start_pos[0] + self.landing_pad_amplitude * np.cos(self.landing_pad_omega * t)
        x = self.landing_pad_base_start_pos[0]
        # y = self.landing_pad_base_start_pos[1] + self.landing_pad_amplitude * np.sin(self.landing_pad_omega * t)
        y = self.landing_pad_base_start_pos[1]
        z = self.landing_pad_base_start_pos[2]
        self.landing_pad_base_pos = [x, y, z]
        p.resetBasePositionAndOrientation(self.landing_pad_id,
                                          self.landing_pad_base_pos,
                                          p.getQuaternionFromEuler([0, 0, 0]),
                                          physicsClientId=self.CLIENT)

    def reset(self):
        """
        환경 리셋:
          - 상위 환경(super) reset 호출
          - 착륙 패드 초기화 및 위치 업데이트
          - 카메라 이미지(초기 프레임)를 받아서 프레임 버퍼를 stack_size만큼 채움
          - stacked observation 반환
        """
        obs = super().reset()

        # 착륙 패드 리셋 및 초기 업데이트
        self._resetLandingPad()
        self._updateLandingPad()

        return obs

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

    def _getDroneImages(self, nth_drone, segmentation: bool=True):
        if self.IMG_RES is None:
            print("[ERROR] in VisionLandingAviary._getDroneImages(), remember to set self.IMG_RES to np.array([width, height])")
            exit()

        # 1) 드론 오리엔테이션(쿼터니언) → 회전행렬
        quat_drone = self.quat[nth_drone, :]
        rot_mat = np.array(p.getMatrixFromQuaternion(quat_drone)).reshape((3,3))

        # 2) 드론에서의 카메라 offset, 보고싶은 방향, up 벡터
        camera_offset = np.array([0., 0., 0.087])        # 드론 중심 대비 카메라 위치
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
            nearVal=0.1,
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

        return {"images": self._get_stacked_obs(rgb), "drone_state": self._getDroneStateVector(nth_drone=0)}
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
                            "drone_state": spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)})
        # return spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

    def _computeReward(self):

        return self._computeReward_backup_thanks_to_Pawel()

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
        distance_x = np.abs(drone_position[0]-UGV_pos[0])
        distance_y = np.abs(drone_position[1]-UGV_pos[1])
        if np.abs(angle) > 30 and (distance_y > 0.8 and distance_x > 0.8):
            combined_reward = -0.01
        return combined_reward

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

        # 너무 빠르게 움직이는 경우
        if abs(drone_z_vel) / self.SPEED_LIMIT[2] > 1.1:
            return 0.0
        # 상승 하는 경우
        if drone_z_vel > 0:
            return -0.1
        # 안전하게 하강하는 경우
        if desired_z_vel < drone_z_vel <= 0:
            return (alpha ** (drone_z_vel / desired_z_vel) - 1) / (alpha - 1)
        # 다소 빠르게 하강하는 경우
        elif drone_z_vel <= desired_z_vel:
            return -0.01
        else:
            raise "VisionLandingAviary env._compute_vertical_velocity_reward(): This should not happen!"

    def _compute_reward_landing_or_crashing(self):
        drone_id = self.DRONE_IDS[0]
        drone_altitude = self.pos[0, 2]

        if drone_altitude >= self.pad_height and p.getContactPoints(bodyA=drone_id, physicsClientId=self.CLIENT) != ():
            print('    VisionLandingAviary env: Landed!')
            return 100.0
        elif drone_altitude < self.pad_height and p.getContactPoints(bodyA=drone_id, physicsClientId=self.CLIENT) != ():
            print('    VisionLandingAviary env: Crashed!')
            return -1.0
        else:
            return 0.0

    def _compute_reward_visibility(self):
        """Returns positive reward if LOS; otherwise, negative reward."""
        pad_position = self._get_pad_center_position()  # numpy (3,)
        if self._check_los(pad_position):
            return 1.0
        else:
            print("    VisionLandingAviary env: Out of LOS!")
            return -0.01

    def _check_los(self, pad_position):
        """
        드론의 카메라 FOV 안에 pad_position(타겟)이 들어왔는지 여부를 True/False로 반환.
        """
        # 드론 위치, 쿼터니언 가져오기
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
        if self.step_counter * self.TIMESTEP >= self.EPISODE_LEN_SEC:
            return True

        return False

    def _computeInfo(self):
        """
        추가 정보 반환:
          - 드론의 현재 위치, 착륙 패드의 위치, 두 대상 간의 수평 거리를 포함
        """
        info = {}
        info['drone_pos'] = self.pos[0]
        info['landing_pad_base_pos'] = self.landing_pad_base_pos
        info['horizontal_distance'] = np.linalg.norm(np.array(self.pos[0][:2]) - np.array(self.landing_pad_base_pos[:2]))
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

    def convert_onboard_images_to_video(self, output_file="onboard_video.mp4", fps=24):
        """
        onboard 이미지가 저장된 폴더(self.ONBOARD_IMG_PATH) 내의 PNG 파일들을 ffmpeg를 이용하여 동영상으로 변환하는 예시 메서드.
        PNG 파일은 "frame_<번호>.png" 형식으로 저장되어 있다고 가정함.
        """
        # TODO: still in alpha
        # ffmpeg 명령어 예시:
        # ffmpeg -y -framerate 24 -i frame_%d.png -c:v libx264 -pix_fmt yuv420p onboard_video.mp4
        cmd = [
            "ffmpeg",
            "-y",  # 기존 파일 덮어쓰기
            "-framerate", str(fps),
            "-i", os.path.join(self.ONBOARD_IMG_PATH, "frame_%d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_file
        ]
        print("Converting onboard images to video...")
        subprocess.run(cmd)
        print("Done. Video saved to:", output_file)

    def convert_external_images_to_video(self, output_file="external_video.mp4", fps=24):
        """
        만약 외부 카메라 이미지가 저장되는 폴더(self.IMG_PATH)에 PNG 파일들이 저장된다면,
        해당 폴더의 PNG 파일들을 ffmpeg로 동영상으로 변환하는 예시 메서드.
        (BaseAviary의 record 옵션에서 DIRECT 모드로 동작 시 사용됨)
        """
        # TODO: still in alpha
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", os.path.join(self.IMG_PATH, "frame_%d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_file
        ]
        print("Converting external images to video...")
        subprocess.run(cmd)
        print("Done. Video saved to:", output_file)