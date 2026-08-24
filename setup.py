"""Setuptools hook for platform-specific wheels containing the Rust server."""

from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel


class BinaryDistribution(Distribution):
    """Mark the wheel as platform-specific even though the control plane is Python."""

    def has_ext_modules(self) -> bool:
        return True


class PlatformWheel(bdist_wheel):
    """Use the Python 3 compatibility tag because no Python extension is shipped."""

    def get_tag(self) -> tuple[str, str, str]:
        _, _, platform_tag = super().get_tag()
        return "py3", "none", platform_tag


setup(distclass=BinaryDistribution, cmdclass={"bdist_wheel": PlatformWheel})
