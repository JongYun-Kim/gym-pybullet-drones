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
    - 헬리패드: 움직이는 착륙 패드 (simple car + helipad texture from parsed_pad.urdf in assets dir)
    - Observation: 드론 카메라에서 촬영한 RGB 이미지(알파 채널 제외)를 4 프레임 stack
    - Action: ActionType.RPM or VEL
    - Reward: working on it...
    - 기록: 환경 외부 카메라 기록(BaseAviary의 record 옵션) 외에도, onboard 카메라 이미지가 PNG로 저장되며,
             후에 ffmpeg를 이용해 동영상으로 변환 가능.
"""
# TODOs:
# (o) rgb to grey scale
# (2) _getDroneImages 메서드 제대로 된건지 확인 하기 (fov 등은 잘 되는데)
# (3) _computeReward* 더 클린 하게 바꾸기 (읽기 좋게좀...; modularize for curriculum learning)
# (4) _computeDone 체크 하기! contact 를 체크 해서 curriculum learning 에 통합 해야함.
# (5) _checkLOS 만들기
# (5) observation stack 의 간격을 체크할 것
# (6) Randomizations
#   (6-1) 착륙 패드 방향 랜덤화(yaw 로 구현 되어 있으나 확인 해 봐야함.)
#   (6-2) 착륙 패드 texture 랜덤화
#   (6-3) 드론 초기 위치 랜덤화 (반드시 vision 범위 내에서 시작할 것)
import os
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import BaseSingleAgentAviary, ObservationType, ActionType
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.BaseAviary import ImageType  # onboard 이미지 저장에 사용
from gym_pybullet_drones.utils.utils import rgb2gray
import subprocess


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
                 ):
        self.stack_size = stack_size  # used in _observationSpace(), which is called in super().__init__()
        self.assets_path = "./gym_pybullet_drones/assets"  # assumes pwd=='ur_project_dir/gym-pybullet-drones/'
        self.fov = fov
        self.channel_first = channel_first

        # 그레이 스케일 사용 여부 (Complicated; but tried to maintain backward compatibility)
        self.use_gray_scale = True if obs == ObservationType.BW else False
        obs = ObservationType.RGB if obs == ObservationType.BW else obs

        # 착륙 패드 관련 파라미터 초기화 (IMG_RES와 무관하므로 먼저 호출 가능)
        self.pad_link_center_idx = None
        self._resetLandingPad()

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
        여기서는 pybullet_data 내의 cube.urdf를 사용하며, globalScaling을 조절하여 착륙 패드 크기를 설정함.
        여기서는 착륙 패드를 'parsed_pad.urdf' 파일을 가져옴.
        내부에는 base1.obj를 참고하고
        그 내부에는 base1.mtl을 참고하며
        그 내부에는 입힐 texture를 image 파일을 지정함. 모두 self.assets_path 내에 있어야 함. 복작복작복잡하네.
        - Body ID: 0, Name: plane  as idk yet
        - Body ID: 1, Name: cf2    as self.DRONE_IDS
        - Body ID: 2, Name: car    as self.landing_pad_id
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
        """
        pad_urdf = self.assets_path + "/parsed_pad.urdf"
        yaw = np.random.uniform(-np.pi/12.0, np.pi/12.0)
        pad_start_orientation_euler = [0,0, yaw]
        pad_start_orientation_quaternion = p.getQuaternionFromEuler(pad_start_orientation_euler)

        print("PyBullet is searching in:", os.getcwd())  # Check current directory
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

        # 카메라에서 RGB 이미지를 받아 알파 채널 포함됨 (shape: [H, W, 4])
        rgb, _, _ = self._getDroneImages(0, segmentation=False)

        # onboard 이미지 저장 (record=True이면)
        if self.RECORD and (self.step_counter % self.IMG_CAPTURE_FREQ == 0):
            self._exportImage(img_type=ImageType.BW if self.use_gray_scale else ImageType.RGB,
                              img_input=rgb,
                              path=self.ONBOARD_IMG_PATH,
                              frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ))

        return self._get_stacked_obs(rgb)

    def _getDroneImages(self, nth_drone, segmentation: bool=True):
        if self.IMG_RES is None:
            print("[ERROR] in VisionLandingAviary._getDroneImages(), remember to set self.IMG_RES to np.array([width, height])")
            exit()

        # 기존의 드론 회전 행렬 관련 코드는 제거하거나 무시
        # rot_mat = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)

        # 카메라 위치: 드론 중심에서 필요에 따라 약간 아래로 배치 (예: [0, 0, 0.0] 또는 [-0.1] 등)
        cameraEye = self.pos[nth_drone, :] + np.array([0, 0, 0.0])

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

        # onboard 이미지 저장 (record=True이면)
        if self.RECORD and (self.step_counter % self.IMG_CAPTURE_FREQ == 0):
            self._exportImage(img_type=ImageType.BW if self.use_gray_scale else ImageType.RGB,
                              img_input=rgb,
                              path=self.ONBOARD_IMG_PATH,
                              frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ))

        return self._get_stacked_obs(rgb)

    def _observationSpace(self):
        """
        Observation space 재정의:
          - 각 카메라 프레임의 크기는 IMG_RES (예: [64, 48])
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

        return spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

    def _computeReward(self):
        """
        Reward 함수 예시:
          - 드론과 착륙 패드 사이의 수평 거리, 고도 차, 수평 속도에 대해 패널티 부여
          - 착륙 성공 (수평거리 < 0.2m, 고도 차 < 0.2m, 수평 속도 < 0.1m/s) 시 큰 보너스 지급
          - 드론이 너무 낮은 상태에서 패드와 멀어졌다면(크래시) 추가 패널티
        """
        # TODO: modularize this reward function
        #       and allow to have custom combinations via config(dict) for curriculum learning
        drone_pos = np.array(self.pos[0])
        pad_pos = np.array(self.landing_pad_base_pos)
        horizontal_distance = np.linalg.norm(drone_pos[:2] - pad_pos[:2])
        vertical_distance = drone_pos[2] - pad_pos[2]
        drone_vel = np.array(self.vel[0])
        horizontal_speed = np.linalg.norm(drone_vel[:2])

        reward = - horizontal_distance - 0.1 * vertical_distance - 0.01 * horizontal_speed

        # 착륙 성공 조건: 충분히 근접하고 낮은 속도라면 보너스 지급
        if horizontal_distance < 0.2 and vertical_distance < 0.2 and horizontal_speed < 0.1:
            reward += 100

        # 크래시 상황: 드론 높이가 매우 낮으면서 패드로부터 멀면 패널티
        if drone_pos[2] < 0.05 and horizontal_distance > 0.5:
            reward -= 100

        return reward

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

    def _computeReward_thanks_to_Pawel(self):
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
        moves_down_and_safe_in_z = (0 > drone_velocity[2]) * (drone_velocity[2] > desired_z_vel)

        # (1) Compute reward: Vertical velocity
        if moves_down_and_safe_in_z:
            reward_z_vel = (alpha**(drone_velocity[2]/desired_z_vel) -1)/(alpha -1)
        else:  # Penalize if drone moves up or too fast
            if abs(drone_velocity[2])/self.SPEED_LIMIT[2] > 1.1:
                reward_z_vel = 0
            else:
                if drone_velocity[2] < desired_z_vel:
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

        drone_id = self.DRONE_IDS[0]
        if drone_position[2] >= 0.275 and p.getContactPoints(bodyA=drone_id, physicsClientId=self.CLIENT) != ():
            print('landed!')
            combined_reward =  140 + combined_reward
        elif drone_position[2]  < 0.275 and p.getContactPoints(bodyA=drone_id, physicsClientId=self.CLIENT) != ():
            print('crashed!')
            combined_reward = -1 #normalized_distance_xy * 10 #0#5*distance_xy + combined_reward
        else:
            combined_reward =  combined_reward
        distance_x = np.abs(drone_position[0]-UGV_pos[0])
        distance_y = np.abs(drone_position[1]-UGV_pos[1])
        if np.abs(angle) > 30 and (distance_y > 0.8 and distance_x > 0.8):
            combined_reward = -0.01
        return combined_reward

    def _computeDone(self):
        """
        종료 조건:
          - 에피소드 시간 초과 (self.EPISODE_LEN_SEC)
          - 드론이 지면에 충돌(높이 <= 0)
        """
        done = False
        drone_pos = np.array(self.pos[0])  # nth_drone=0
        if self.step_counter * self.TIMESTEP >= self.EPISODE_LEN_SEC:
            done = True
        if drone_pos[2] <= 0.0:
            done = True
        return done

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