from glob import glob
from setuptools import find_packages, setup

package_name = 'exoswarm_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yubo Wang',
    maintainer_email='stanleyhoo042707@gmail.com',
    description='Super-Earth world generation, Gazebo Harmonic integration, mock world and sensor effects for the Redshift fleet',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mock_world_node = exoswarm_sim.mock_world_node:main',
            'world_interface_node = exoswarm_sim.world_interface_node:main',
            'sensor_effects_node = exoswarm_sim.sensor_effects_node:main',
        ],
    },
)
