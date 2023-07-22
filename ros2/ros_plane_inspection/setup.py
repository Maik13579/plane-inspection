from setuptools import setup

package_name = 'ros_plane_inspection'

data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ]

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Maik Knof',
    maintainer_email='maik.knof@gmx.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'ros_plane_inspection_node = ros_plane_inspection.ros_plane_inspection_node:main',
        ],
    },
)
