from hydra.utils import get_class


class BaseTasksMixin:
    def __init__(self, *args, **kwargs):

        task_config = kwargs["task_config"]
        task_target_cls = get_class(task_config["_target_"])

        if len(self.__class__.__bases__) > 1:
            self.__class__.__bases__ = (self.__class__.__bases__[0],)
        self.__class__.__bases__ += (task_target_cls,)
        task_target_cls.__init__(self, **task_config)
