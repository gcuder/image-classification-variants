import re
import typing
import tensorflow as tf
from abc import ABCMeta
from abc import abstractmethod as abstract_method
from collections import OrderedDict
from inspect import isclass as is_class

# Camel case to snake case utils
_first_cap_re = re.compile("(.)([A-Z][a-z0-9]+)")
_all_cap_re = re.compile("([a-z0-9])([A-Z])")


class RegistryKeyError(KeyError):
    """
    Used to differentiate a registry lookup from a standard KeyError.
    This is especially useful when a registry class expects to extract
    values from dicts to generate keys.
    """

    pass


class BaseRegistry(typing.Mapping, metaclass=ABCMeta):
    """
    Base functionality for registries.
    """

    def __contains__(self, key):
        """
        Returns whether the specified key is registered.
        """
        try:
            # Use :py:meth:`get_class` instead of :py:meth:`__getitem__`
            # in case the registered class' initializer requires
            # arguments.
            self.get_class(key)
        except RegistryKeyError:
            return False
        else:
            return True

    def __dir__(self):
        return list(self.keys())

    def __getitem__(self, key):
        """
        Shortcut for calling :py:meth:`get` with empty args/kwargs.
        """
        return self.get(key)

    def __iter__(self):
        """
        Returns a generator for iterating over registry keys, in the
        order that they were registered.
        """
        return self.keys()

    @abstract_method
    def __len__(self):
        """
        Returns the number of registered classes.
        """
        raise NotImplementedError(
            "Not implemented in {cls}.".format(cls=type(self).__name__),
        )

    def __missing__(self, key):
        """
        Returns the result (or raises an exception) when the requested
        key can't be found in the registry.
        """
        raise RegistryKeyError(key)

    @abstract_method
    def get_class(self, key):
        """
        Returns the class associated with the specified key.
        """
        raise NotImplementedError(
            "Not implemented in {cls}.".format(cls=type(self).__name__),
        )

    def get(self, key, *args, **kwargs):
        """
        Creates a new instance of the class matching the specified key.
        :param key:
            The corresponding load key.
        :param args:
            Positional arguments passed to class initializer.
            Ignored if the class registry was initialized with a null
            template function.
        :param kwargs:
            Keyword arguments passed to class initializer.
            Ignored if the class registry was initialized with a null
            template function.
        References:
          - :py:meth:`__init__`
        """
        return self.create_instance(self.get_class(key), *args, **kwargs)

    @staticmethod
    def gen_lookup_key(key: typing.Any) -> typing.Hashable:
        """
        Used by :py:meth:`get` to generate a lookup key.
        You may override this method in a subclass, for example if you
        need to support legacy aliases, etc.
        """
        return key

    @staticmethod
    def create_instance(class_: type, *args, **kwargs):
        """
        Prepares the return value for :py:meth:`get`.
        You may override this method in a subclass, if you want to
        customize the way new instances are created.
        :param class_:
            The requested class.
        :param args:
            Positional keywords passed to :py:meth:`get`.
        :param kwargs:
            Keyword arguments passed to :py:meth:`get`.
        """
        return class_(*args, **kwargs)

    @abstract_method
    def items(self) -> typing.Iterable[typing.Tuple[typing.Hashable, type]]:
        """
        Iterates over registered classes and their corresponding keys,
        in the order that they were registered.
        Note: For compatibility with Python 3, this method should
        return a generator.
        """
        raise NotImplementedError(
            "Not implemented in {cls}.".format(cls=type(self).__name__),
        )

    def keys(self) -> typing.Iterable[typing.Hashable]:
        """
        Returns a generator for iterating over registry keys, in the
        order that they were registered.
        Note: For compatibility with Python 3, this method should
        return a generator.
        """
        for item in self.items():
            yield item[0]

    def values(self) -> typing.Iterable[type]:
        """
        Returns a generator for iterating over registered classes, in
        the order that they were registered.
        Note: For compatibility with Python 3, this method should
        return a generator.
        """
        for item in self.items():
            yield item[1]


class MutableRegistry(
    BaseRegistry,
    typing.MutableMapping,
    metaclass=ABCMeta,
):
    """
    Extends :py:class:`BaseRegistry` with methods that can be used to
    modify the registered classes.
    """

    def __init__(self, registry_name: typing.Optional[str] = None) -> None:
        """
        :param registry_name:
            If provided, :py:meth:`register` will automatically detect
            the key to use when registering new classes.
        """
        super(MutableRegistry, self).__init__()

        self.registry_name = registry_name

    def __delitem__(self, key: typing.Hashable) -> None:
        """
        Provides alternate syntax for un-registering a class.
        """
        self._unregister(key)

    def __repr__(self):
        return "{type}({attr_name!r})".format(
            attr_name=self.registry_name,
            type=type(self).__name__,
        )

    def __setitem__(self, key: str, class_: type) -> None:
        """
        Provides alternate syntax for registering a class.
        """
        self._register(key, class_)

    def register(self, key):
        """
        Decorator that registers a class with the registry.
        Example::
           registry = ClassRegistry(attr_name='widget_type')
           @registry.register
           class CustomWidget(BaseWidget):
             widget_type = 'custom'
             ...
           # Override the registry key:
           @registry.register('premium')
           class AdvancedWidget(BaseWidget):
             ...
        :param key:
            The registry key to use for the registered class.
            Optional if the registry's :py:attr:`attr_name` is set.
        """
        # if key is None:

        if is_class(key):
            # Note that ``getattr`` will raise an AttributeError if
            # the class doesn't have the required attribute.
            self._register(get_default_name(key), key)
            return key

        def _decorator(cls):
            self._register(key, cls)
            return cls

        return _decorator

    def unregister(self, key: typing.Any) -> type:
        """
        Unregisters the class with the specified key.
        :param key:
            The registry key to remove (not the registered class!).
        :return:
            The class that was unregistered.
        :raise:
            - :py:class:`KeyError` if the key is not registered.
        """
        return self._unregister(self.gen_lookup_key(key))

    @abstract_method
    def _register(self, key: typing.Hashable, class_: type) -> None:
        """
        Registers a class with the registry.
        """
        raise NotImplementedError(
            "Not implemented in {cls}.".format(cls=type(self).__name__),
        )

    @abstract_method
    def _unregister(self, key: typing.Hashable) -> type:
        """
        Unregisters the class at the specified key.
        """
        raise NotImplementedError(
            "Not implemented in {cls}.".format(cls=type(self).__name__),
        )


class ClassRegistry(MutableRegistry):
    """
    Maintains a registry of classes and provides a generic factory for
    instantiating them.
    """

    def __init__(
            self,
            registry_name: typing.Optional[str] = None,
            unique: bool = False,
    ) -> None:
        """
        :param registry_name:
            Name of the registry
        :param unique:
            Determines what happens when two classes are registered with
            the same key:
            - ``True``: The second class will replace the first one.
            - ``False``: A ``ValueError`` will be raised.
        """
        super(ClassRegistry, self).__init__(registry_name)

        self.unique = unique

        self._registry = OrderedDict()

    def __len__(self):
        """
        Returns the number of registered classes.
        """
        return len(self._registry)

    def __repr__(self):
        return "{type}(attr_name={attr_name!r}, unique={unique!r})".format(
            attr_name=self.registry_name,
            type=type(self).__name__,
            unique=self.unique,
        )

    def get_class(self, key):
        """
        Returns the class associated with the specified key.
        """
        lookup_key = self.gen_lookup_key(key)

        try:
            return self._registry[lookup_key]
        except KeyError:
            return self.__missing__(lookup_key)

    def items(self) -> typing.Iterable[typing.Tuple[typing.Hashable, str]]:
        """
        Iterates over all registered classes, in the order they were
        added.
        """
        return self._registry.items()

    def _register(self, key: typing.Hashable, class_: type) -> None:
        """
        Registers a class with the registry.
        """
        if key in ["", None]:
            raise ValueError(
                "Attempting to register class {cls} "
                "with empty registry key {key!r}.".format(
                    cls=class_.__name__,
                    key=key,
                ),
            )

        if self.unique and (key in self._registry):
            raise RegistryKeyError(
                "{cls} with key {key!r} is already registered.".format(
                    cls=class_.__name__,
                    key=key,
                ),
            )

        self._registry[key] = class_

    def _unregister(self, key: typing.Hashable) -> type:
        """
        Unregisters the class at the specified key.
        """
        return (
            self._registry.pop(key) if key in self._registry else self.__missing__(key)
        )


def get_default_name(cls: [type, str]) -> str:
    """Convert a class name to the registry's default name for the class.
    Args:
      cls: Object class.
    Returns:
      The registry's default name for the class.
    """
    if isinstance(cls, str):
        return convert_camel_to_snake(cls)
    return convert_camel_to_snake(cls.__name__)


def convert_camel_to_snake(name: str) -> str:
    s1 = _first_cap_re.sub(r"\1_\2", name)
    return _all_cap_re.sub(r"\1_\2", s1).lower()


backbone_registry = ClassRegistry(registry_name='backbones')


def backbones(key: str, **kwargs) -> tf.keras.layers.Layer:
    """Calls a layer from the registry.

    If kwargs do not contain additional arguments, the object is
    returned uninitialized.

    :param key: Name of the registered object
    :param kwargs: Additional arguments
    :return: Keras layer
    """
    if len(kwargs) == 0:
        return backbone_registry.get_class(key)
    return backbone_registry.get(key=key, **kwargs)
