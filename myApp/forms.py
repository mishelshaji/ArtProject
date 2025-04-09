from django import forms
from .models import *
import uuid

class TeacherRequestForm(forms.ModelForm):
    class Meta:
        model = TeacherRequest
        fields = ['full_name', 'email', 'phone_number', 'qualification', 'experience_years', 'skills', 'resume']


class ContactUsForm(forms.ModelForm):
    class Meta:
        model = ContactUs
        fields = '__all__'
        widgets = {
            'first_name': forms.TextInput(attrs={'class': 'form-control'}),
            'last_name': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
            'phone_number': forms.TextInput(attrs={'class': 'form-control'}),
            'message': forms.Textarea(attrs={'class': 'form-control'}),
        }

class CourseCreateForm(forms.ModelForm):
    class Meta:
        model = Course
        fields = ['name', 'short_description', 'description', 'price', 'sub_category', 'duration', 'learning_outcome', 'featured_image']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'short_description': forms.Textarea(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control'}),
            'price': forms.NumberInput(attrs={'class': 'form-control'}),
            'instructor': forms.Select(attrs={'class': 'form-control'}),
            'sub_category': forms.Select(attrs={'class': 'form-control'}),
            'duration': forms.TextInput(attrs={'class': 'form-control'}),
            'learning_outcome': forms.Textarea(attrs={'class': 'form-control'}),
            'featured_image': forms.FileInput(attrs={'class': 'form-field'}),
        }


class CourseLessonForm(forms.ModelForm):
    subtitle = forms.CharField(widget=forms.Textarea(attrs={'class': 'form-control'}), help_text='Please use WEBVTT format', required=False)
    class Meta:
        model = CourseLesson
        fields = ['title', 'description', 'resource']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control'}),
            'resource': forms.FileInput(attrs={'class': 'form-field'}),
        }


class ProductCreateForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = ['name', 'short_description', 'description', 'price', 'image', 'stock', 'category']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'short_description': forms.Textarea(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control'}),
            'price': forms.NumberInput(attrs={'class': 'form-control'}),
            'image': forms.FileInput(attrs={'class': 'form-field'}),
            'stock': forms.NumberInput(attrs={'class': 'form-control'}),
            'category': forms.Select(attrs={'class': 'form-control'}),
        }

class OrderForm(forms.ModelForm):
    class Meta:
        model = Order
        fields = ['address']
        widgets = {
            'address': forms.Textarea(attrs={'class': 'form-control'}),
        }

class CreateEventForm(forms.ModelForm):
    class Meta:
        model = Event
        fields = '__all__'
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control'}),
            'start_time': forms.TextInput(attrs={'class': 'form-control', 'type': 'datetime-local'}),
            'end_time': forms.TextInput(attrs={'class': 'form-control', 'type': 'datetime-local'}),
            'category': forms.Select(attrs={'class': 'form-control'}),
            'total_seats': forms.NumberInput(attrs={'class': 'form-control'}),
            'banner_image': forms.FileInput(attrs={'class': 'form-field'}),
            'promo_video_url': forms.URLInput(attrs={'class': 'form-control'}),
            'organizer_name': forms.TextInput(attrs={'class': 'form-control'}),
            'registration_url': forms.URLInput(attrs={'class': 'form-control'}),
            'location_url': forms.URLInput(attrs={'class': 'form-control'}),
            'entry_fee': forms.NumberInput(attrs={'class': 'form-control'}),
            'organizer_phone': forms.TextInput(attrs={'class': 'form-control'}),
            'organizer_email': forms.EmailInput(attrs={'class': 'form-control'}),
            'instructions': forms.Textarea(attrs={'class': 'form-control'}),
        }

class TeacherCreateLiveClassForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        self.fields['course'].queryset = Course.objects.filter(instructor=user)
        self.fields['live_url'].initial = f'https://meet.jit.si/{uuid.uuid4()}'

    class Meta:
        model = LiveClass
        fields = '__all__'
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control'}),
            'remark': forms.Textarea(attrs={'class': 'form-control'}),
            'start_time': forms.TextInput(attrs={'class': 'form-control', 'type': 'datetime-local'}),
            'end_time': forms.TextInput(attrs={'class': 'form-control', 'type': 'datetime-local'}),
            'course': forms.Select(attrs={'class': 'form-control'}),
            'live_url': forms.URLInput(attrs={'class': 'form-control'}),
        }