from setuptools import setup
import os
from glob import glob

package_name = 'disaster_brain'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kayaclaridge-sys',
    maintainer_email='kaya.claridge@gmail.com',
    description='Disaster Rover brain + Gemma 4 mission agent',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'brain_node = disaster_brain.brain_node:main',
            'mission_agent = disaster_brain.mission_agent:main',
        ],
    },
)
