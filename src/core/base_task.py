from hydra.utils import get_class

class BaseTasksMixin:
    def __init__(self, *args, **kwargs):

        task_target_cls = get_class(kwargs["task_target"])

        if len(self.__class__.__bases__) > 1:
            self.__class__.__bases__ = (self.__class__.__bases__[0],)
        self.__class__.__bases__ += (task_target_cls,)
        task_target_cls.__init__(self, *args, **kwargs)