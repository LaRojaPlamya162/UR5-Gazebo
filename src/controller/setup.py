from setuptools import find_packages, setup
from glob import glob
import os
package_name = 'controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['bc_model_v2.pth'])
        #(os.path.join('share', package_name, 'models'), ['bc_model_v2.pth']),
        #('share/' + package_name, ['bc_model_v2.pth'])
        #(os.path.join('share', package_name, 'models'), glob('my_controller/bc_model.pth')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nta',
    maintainer_email='nta@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'controller = controller.controller:main',
        ],
    },
)