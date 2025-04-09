from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse, reverse_lazy
from django.contrib.auth.decorators import login_required
from django.views.generic import CreateView, UpdateView, ListView
from django.contrib.auth.mixins import LoginRequiredMixin
from ..models import Category, SubCategory
from ..forms import *
from openai import OpenAI

@login_required
def teacher_home(request):
    context = {}
    context['courses'] = Course.objects.filter(instructor=request.user)
    courses = Course.objects.prefetch_related('feedback').filter(instructor=request.user)

    if(context['courses'].count() == 0):
        return redirect(reverse('teacher_pick_category'))
    
    context['categories'] = Category.objects.all()
    return render(request, 'teacher_home.html', context)

def teacher_pick_category(request):
    context = {}
    context['categories'] = Category.objects.all()
    return render(request, 'teacher_pick_category.html', context)   

@login_required
def teacher_pick_subcategory(request, id):
    context = {}
    context['subcategories'] = SubCategory.objects.filter(category_id=id)
    return render(request, 'teacher_pick_subcategory.html', context)

def teacher_create_course(request, id):
    sub_category = get_object_or_404(SubCategory, id=id)
    category = sub_category.category

    context = {}
    if request.method == 'GET':
        context['form'] = CourseCreateForm(initial={'sub_category': sub_category})
        return render(request, 'teacher_create_course.html', context)

    if request.method == 'POST':
        form = CourseCreateForm(data=request.POST, files=request.FILES, initial={'sub_category': sub_category})

        if form.is_valid():
            course = form.save(commit=False)
            course.instructor = request.user
            course.save()
            return redirect('teacher_home')

        context['form'] = form
        return render(request, 'teacher_create_course.html', context)
    
@login_required
def teacher_edit_course(request, id):
    course = get_object_or_404(Course, id=id, instructor=request.user)
    context = {}

    if request.method == 'GET':
        context['form'] = CourseCreateForm(instance=course)
        return render(request, 'teacher_edit_course.html', context)

    elif request.method == 'POST':
        form = CourseCreateForm(data=request.POST, files=request.FILES, instance=course)

        if form.is_valid():
            course = form.save()
            return redirect('teacher_home')

        context['form'] = form
        return render(request, 'teacher_edit_course.html', context)
    
@login_required
def teacher_delete_course(request, id):
    course = get_object_or_404(Course, id=id, instructor=request.user)
    course.status = 'archived'
    course.save()
    return redirect('teacher_home')

@login_required
def teacher_restore_course(request, id):
    course = get_object_or_404(Course, id=id, instructor=request.user)
    course.status = 'draft'
    course.save()
    return redirect('teacher_home')

@login_required
def teacher_request_course_approval(request, id):
    course = get_object_or_404(Course, id=id, instructor=request.user)
    course.status = 'pending'
    course.save()
    return redirect('teacher_home')

@login_required
def teacher_create_course_lesson(request, id):
    course = get_object_or_404(Course, id=id, instructor=request.user)
    context = {}
    context['lessons'] = CourseLesson.objects.filter(course=course)
    to_edit_id = request.GET.get('edit')
    to_edit = CourseLesson.objects.filter(id=to_edit_id).first()

    if request.method == 'GET':
        context['form'] = CourseLessonForm(instance=to_edit)
        return render(request, 'teacher_create_course_lesson.html', context)
    
    elif request.method == 'POST':
        form = CourseLessonForm(data=request.POST, files=request.FILES, instance=to_edit)

        if form.is_valid():
            lesson = form.save(commit=False)
            lesson.course = course
            lesson.save()
            if form.cleaned_data['subtitle'] != None:
                subtitle = form.cleaned_data['subtitle']
                lesson_video_subtitles = []
                client = OpenAI(api_key=settings.OPENAI_KEY)
                
                for i in LessonVideoSubtitle.LANGUAGE_CHOICES:
                    completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system", 
                            "content": f"You have to translate the WEBVTT file content into {i[0]}. Do not generate any extra text."
                        },
                        {
                            "role": "user",
                            "content": subtitle
                        }
                    ])

                    language = i[0]
                    subtitle_obj = LessonVideoSubtitle()
                    subtitle_obj.language = language
                    subtitle_obj.subtitle = completion.choices[0].message.content
                    print(subtitle_obj.subtitle)
                    subtitle_obj.lesson = lesson
                    lesson_video_subtitles.append(subtitle_obj)
                LessonVideoSubtitle.objects.bulk_create(lesson_video_subtitles)

                # Creating a summary
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system", 
                            "content": f"You have to summarize the WEBVTT file content. Do not generate any extra text."
                        },
                        {
                            "role": "user",
                            "content": subtitle
                        }
                    ]
                )
                lesson.summary = completion.choices[0].message.content
                lesson.save()
            return redirect(reverse('teacher_create_course_lesson', args=[course.id]))

        context['form'] = form
        return render(request, 'teacher_create_course_lesson.html', context)

@login_required
def teacher_delete_course_lesson(request, id):
    lesson = get_object_or_404(CourseLesson, id=id, course__instructor=request.user)
    course = lesson.course
    lesson.delete()
    return redirect(reverse('teacher_create_course_lesson', args=[course.id]))

@login_required
def teacher_course_students(request, id):
    course = get_object_or_404(Course, id=id)
    context = {}
    context['students'] = EnrolledCourse.objects.filter(course=course)
    context['course'] = course
    return render(request, 'teacher_course_students.html', context)

@login_required
def teacher_list_enrolled_students(request):
    context = {}
    context['students'] = EnrolledCourse.objects.all(course__instructor=request.user)
    return render(request, 'teacher_list_enrolled_students.html', context)

class LiveClassListView(LoginRequiredMixin, ListView):
    model = LiveClass
    template_name = 'teacher_list_liveclass.html'

    def get_queryset(self):
        return super().get_queryset().filter(course__instructor=self.request.user)

class LiveClassCreateView(LoginRequiredMixin, CreateView):
    model = LiveClass
    form_class = TeacherCreateLiveClassForm
    template_name = 'teacher_create_liveclass.html'
    success_url = reverse_lazy('teacher_home')

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['user'] = self.request.user
        return kwargs

    def form_valid(self, form):
        course = form.cleaned_data['course']
        students = EnrolledCourse.objects.filter(course=course)
        for i in students:
            id = i.student_id
            notification = Notification()
            notification.content = f'A live class has been scheduled for {course.name}. Please join the class.'
            notification.user_id = id
            notification.save()
        return super().form_valid(form)

def teacher_delete_liveclass(request, id):
    liveclass = get_object_or_404(LiveClass, id=id, course__instructor=request.user)
    liveclass.delete()
    return redirect('teacher_liveclass')