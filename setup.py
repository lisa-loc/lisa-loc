import os
from glob import glob
from setuptools import find_packages, setup 

package_name = 'lisa_loc'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.xml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sumit Gore',
    maintainer_email='hello@sumietgore.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "lidar_test = lisa_loc.lidar_test:main",
            "ground_extractor = lisa_loc.ground_extractor:main",
            "plane_extractor = lisa_loc.plane_extractor:main",
        ],
    },
)
