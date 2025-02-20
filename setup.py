from setuptools import setup, find_packages

setup(name='gym_pybullet_drones',
    version='1.0.0',
    packages=find_packages(include=["gym_pybullet_drones", "gym_pybullet_drones.*"]),
    install_requires=[
        'numpy==1.23.4',
        'Pillow==9.3.0',
        'matplotlib==3.6.2',
        'cycler==0.11.0',
        'gym==0.23.1',
        'pybullet==3.2.5',
        # 'stable_baselines3==1.6.2',
        'ray[rllib]==2.1.0',
        'scipy==1.9.3',
        'tensorboard==2.19.0',
    ]
)