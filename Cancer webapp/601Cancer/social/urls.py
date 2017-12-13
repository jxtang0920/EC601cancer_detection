from django.conf.urls import include, url
from social import views

from rest_framework.routers import DefaultRouter

router = DefaultRouter()
#router.register(r'profiles', views.ProfileViewSet)
router.register(r'members', views.MemberViewSet)

urlpatterns = [
    # main page
    url(r'^$', views.index),
    # signup page
    url(r'^signup/$', views.signup),
    # register new user
    url(r'^register/$', views.register),
    # login page
    url(r'^login/$', views.login),
    # logout page
    url(r'^logout/$', views.logout),
    # members page
    url(r'^member/$', views.member),
    # friends page
    #url(r'^friends/$', views.friends),
    # user profile edit page
    #url(r'^profile/$', views.profile),
    # Ajax: check if user exists
    url(r'^checkuser/$', views.checkuser),
    # messages page
    #url(r'^messages/$', views.messages),
    # REST api
    url(r'^api/', include(router.urls)),
    # Test view
    url(r'^test/', views.test),
]
