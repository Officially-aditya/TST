from .base import BaseService
from .helpers import work as do_work


class Service(BaseService):
    def run(self) -> int:
        return do_work(1)
