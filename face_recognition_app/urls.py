# your_app/urls.py
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('', views.class_view, name='class'),
    path('student/',views.student_view,name='student'),
    path('studentloginprocess/', views.student_login_process, name='student_login'),
    path('admins/', views.admins_view, name='admin'),
     path('index/', views.index_view, name='index'),
    path('about/', views.about_view, name='about'),
    path('face/', views.face_view, name='face'),
    path('adminlogin/', views.adminlogin_view, name='adminlogin'),
    path('studentlogin/', views.studentlogin_view, name='studentlogin'),
    path('take-attendance/', views.take_attendance_view, name='takeattendance'),
    path('stop-attendance/', views.stop_attendance, name='stop_attendance'),
    path('submit_letter/', views.submit_letter_view, name='submit_letter'),
    path('check-attendance/', views.check_attendance, name='checkattendance'),
    path('view-attendance/', views.view_attendance, name='viewattendance'),
    path('attendance/', views.attendance_view, name='attendance'),
    path('attendance-result/', views.attendance_result_view, name='attendanceresult'),
    path('students-request/', views.students_request_view, name='studentsrequest'),
    path('view-students/', views.view_students_view, name='viewstudents'),
    path('files', views.list_files, name='list_files'),
    path('files/<str:filename>', views.get_file_content, name='get_file_content'),
    path('files/<str:filename>/status/', views.update_status, name='update_status'),
    path('files/', views.get_file_list, name='file_list'),
    path('files/<str:file_name>/', views.get_file_status, name='file_status'),
    path('requeststatus/', views.request_status_view, name='requeststatus'),
    path('add-student/', views.add_student, name='add_student'),
    
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
