from setuptools import setup

package_name = 'my_neuro_arm_goal'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/my_neuro_arm_goal.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Andrea',        
    maintainer_email='awalkerperez@gmail.com', 
    description='My Neuro Arm Goal Python package',
    license='TODO: License declaration', 
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_neuro_arm_goal = my_neuro_arm_goal.my_neuro_arm_goal_node:main', 
        ],
    },
)

