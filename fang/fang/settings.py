"""
Django settings for fang project.

Generated by 'django-admin startproject' using Django 2.1.

For more information on this file, see
https://docs.djangoproject.com/en/2.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.1/ref/settings/
"""

import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Quick-start development settings - unsuitable for production
# 请注意下面这个链接 该链接提供了部署项目的必要参考加 强烈建议阅读一下
# See https://docs.djangoproject.com/en/2.1/howto/deployment/checklist/

# 请注意这里的提示 在上线产品中这个SECRET_KEY是需要保护起来的
# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'uq7##@ciddcuz!-6qum!vgh_f-rc2!k2#ql867tm4!=zy+yx0#'
# SECRET_KEY = os.environ['SECRET_KEY']


# 注意这里的提示 在上线产品中一定不能够将DEBUG设置为True
# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False
ALLOWED_HOSTS = ['*']

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'common',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'api.middlewares.block_sms_middleware',
]

# 配置跨域白名单
# CORS_ORIGIN_WHITELIST = ()
# CORS_ALLOW_CREDENTIALS = True

ROOT_URLCONF = 'fang.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'fang.wsgi.application'


# Database
# 下面的链接是官方文档关于数据库配置的参考 强烈建议阅读一下
# https://docs.djangoproject.com/en/2.1/ref/settings/#databases
# db_user = os.environ['DB_USER']
# db_pass = os.environ['DB_PASS']

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'fang',
        'HOST': '120.77.222.217',
        'PORT': 3306,
        'USER': 'root',
        'PASSWORD': '123456',
        'TIME_ZONE': 'Asia/Chongqing',
    },
    # 权限数据库配置
    'auth_db': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'auth_db',
        'HOST': '120.77.222.217',
        'PORT': 3306,
        'USER': 'root',
        'PASSWORD': '123456',
        'TIME_ZONE': 'Asia/Chongqing',
    },
    # 从数据库1配置
    'slave1': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'fang',
        'HOST': '120.77.222.217',
        'PORT': 3307,
        'USER': 'root',
        'PASSWORD': '123456',
        'TIME_ZONE': 'Asia/Chongqing',
    },
    # 从数据库2配置
    # 'slave2': {
    #     'ENGINE': 'django.db.backends.mysql',
    #     'NAME': 'fang',
    #     'HOST': '120.77.222.217',
    #     'PORT': 3308,
    #     'USER': 'root',
    #     'PASSWORD': '123456',
    #     'TIME_ZONE': 'Asia/Chongqing',
    # }
}

# 配置数据库路由
DATABASE_ROUTERS = [
    'common.db_routers.AuthAdminDbRouter',
    'common.db_routers.MasterSlaveDbRouter',
]

# Password validation
# https://docs.djangoproject.com/en/2.1/ref/settings/#auth-password-validators
"""
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]
"""


# Internationalization
# 国际化配置
# https://docs.djangoproject.com/en/2.1/topics/i18n/

LANGUAGE_CODE = 'zh-hans'

TIME_ZONE = 'Asia/Shanghai'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# 静态文件配置
# https://docs.djangoproject.com/en/2.1/howto/static-files/

STATIC_URL = '/static/'
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]

STATIC_ROOT = '/root/static/'

# MEDIA_ROOT = '/var/www/jackfrued.xyz/media/'
# MEDIA_URL = '/media/'

# 也可以使用第三方云存储来保存静态资源 如七牛、LeanCloud、AWS、OSS等

# 配置基于Memcached/Redis的缓存系统
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': [
            'redis://120.77.222.217:6379/0',
        ],
        'KEY_PREFIX': 'fang',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 500,
            },
            'PASSWORD': '1qaz2wsx',
        }
    },
    'page': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': [
            'redis://120.77.222.217:6379/1',
            'redis://120.77.222.217:6380/1',
            'redis://120.77.222.217:6381/1',
            'redis://120.77.222.217:6382/1',
        ],
        'KEY_PREFIX': 'fang:page',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 1000,
            },
            'PASSWORD': '1qaz2wsx',
        }
    },
    'session': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': [
            'redis://120.77.222.217:6379/2',
            'redis://120.77.222.217:6380/2',
            'redis://120.77.222.217:6381/2',
            'redis://120.77.222.217:6382/2',
        ],
        'KEY_PREFIX': 'fang:session',
        'TIMEOUT': 1209600,
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 1000,
            },
            'PASSWORD': '1qaz2wsx',
        }
    },
    'code': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': [
            'redis://120.77.222.217:6379/3',
            'redis://120.77.222.217:6380/3',
            'redis://120.77.222.217:6381/3',
            'redis://120.77.222.217:6382/3',
        ],
        'KEY_PREFIX': 'fang:code:tel',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 500,
            },
            'PASSWORD': '1qaz2wsx',
        }
    },
}

# 配置将Session放置在缓存中
SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'session'

# 配置Django-REST-Framework
REST_FRAMEWORK = {
    'PAGE_SIZE': 5,
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'EXCEPTION_HANDLER': 'api.exceptions.exception_handler',
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.FormParser',
        'rest_framework.parsers.MultiPartParser',
    ],
    # 'DEFAULT_THROTTLE_CLASSES': [],
    # 'DEFAULT_PERMISSION_CLASSES': (
    #     'rest_framework.permissions.IsAuthenticated',
    # ),
    # 'DEFAULT_AUTHENTICATION_CLASSES': (
    #     'rest_framework_jwt.authentication.JSONWebTokenAuthentication',
    # ),
}

# 配置DRF扩展来支持缓存API接口调用结果
REST_FRAMEWORK_EXTENSIONS = {
    'DEFAULT_CACHE_RESPONSE_TIMEOUT': 150,
    'DEFAULT_USE_CACHE': 'default',
}

# 配置Session的序列化方式 默认是JsonSerializer
# SESSION_SERIALIZER = 'django.contrib.sessions.serializers.PickleSerializer'

# 配置异步任务调度器Celery
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_ACCEPT_CONTENT = ['json',]
CELERY_TIMEZONE = TIME_ZONE

# 配置自动追加斜杠
APPEND_SLASH = True

# 配置日志
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    # 配置日志格式化器
    'formatters': {
        'simple': {
            'format': '%(asctime)s %(module)s.%(funcName)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'verbose': {
            'format': '%(asctime)s %(levelname)s [%(process)d-%(threadName)s] '
                      '%(module)s.%(funcName)s line %(lineno)d: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        }
    },
    # 配置日志过滤器
    'filters': {
        'require_debug_true': {
            '()': 'django.utils.log.RequireDebugTrue',
        },
    },
    # 配置日志处理器
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'filters': ['require_debug_true'],
            'formatter': 'simple',
        },
        'file1': {
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'filename': 'fang.log',
            'when': 'W0',
            'backupCount': 12,
            'formatter': 'simple',
            'level': 'INFO',
        },
        'file2': {
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'filename': 'error.log',
            'when': 'D',
            'backupCount': 31,
            'formatter': 'verbose',
            'level': 'ERROR',
        },
    },
    # 配置日志器
    'loggers': {
        'django': {
            'handlers': ['console', 'file1', 'file2'],
            'propagate': True,
            'level': 'INFO',
        },
    }
}

# 安全相关配置

# 保持HTTPS连接的时间
SECURE_HSTS_SECONDS = 3600
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True

# 自动重定向到安全连接
SECURE_SSL_REDIRECT = True

# 避免浏览器自作聪明推断内容类型
SECURE_CONTENT_TYPE_NOSNIFF = True

# 避免跨站脚本攻击
SECURE_BROWSER_XSS_FILTER = True

# COOKIE只能通过HTTPS进行传输
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True

# 防止点击劫持攻击手段 - 修改HTTP协议响应头
# 当前网站是不允许使用<iframe>标签进行加载的
X_FRAME_OPTIONS = 'DENY'

