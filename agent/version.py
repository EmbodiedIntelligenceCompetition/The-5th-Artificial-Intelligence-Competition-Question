
_MAJOR_VERSION = '0'
_MINOR_VERSION = '3'
_PATCH_VERSION = '0'

_DEV_SUFFIX = 'dev'
_REL_SUFFIX = 'rc0'


__version__ = '.'.join([
    _MAJOR_VERSION,
    _MINOR_VERSION,
    _PATCH_VERSION,
])
__dev_version__ = '{}.{}'.format(__version__, _DEV_SUFFIX)
__rel_version__ = '{}.{}'.format(__version__, _REL_SUFFIX)
