from glob import glob
from setuptools import find_packages, setup

package_name = 'redshift_autonomy'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yubo Wang',
    maintainer_email='stanleyhoo042707@gmail.com',
    description='Local perception, mapping, gravity-aware planning and failure recovery for the Redshift fleet',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = redshift_autonomy.perception_node:main',
            'mapping_coordinator_node = redshift_autonomy.mapping_coordinator_node:main',
            'agent_planner_node = redshift_autonomy.agent_planner_node:main',
            'fleet_brain_node = redshift_autonomy.fleet_brain_node:main',
        ],
    },
)
