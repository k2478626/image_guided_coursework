import glob
from setuptools import setup

package_name = 'my_neuro_arm_goal'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Andrea',
    description='A custom ROS 2 Python package to plan motions for neuro arm using MoveItPy',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_robot_goal = my_neuro_arm_goal.my_robot_goal:main',
        ],
    },
)
