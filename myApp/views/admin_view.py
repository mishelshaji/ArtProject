from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse, reverse_lazy
from django.contrib.auth.decorators import login_required
from django.views.generic import CreateView, UpdateView, ListView
from django.contrib.auth.mixins import LoginRequiredMixin
from ..forms import *
from ..models import *

@login_required
def admin_view(request):
    context = {}
    context['teacher_count'] = User.objects.filter(usertype='teacher').count()
    context['student_count'] = User.objects.filter(usertype='student').count()
    context['pending_teacher_count'] = TeacherRequest.objects.filter(status='pending').count()

    return render(request, 'adminindex.html', context)


@login_required
def admin_contact_us_view(request):
    context = {}
    context['data'] = ContactUs.objects.all()

    return render(request, 'admin_contact_us.html', context)


@login_required
def admin_delete_contact_us_view(request, id):
    contact = ContactUs.objects.get(id=id)
    contact.delete()
    return redirect('admin_contact')

@login_required
def admin_courses_view(request):
    context = {}
    # Get courses with status approved and pending
    context['courses'] = Course.objects.filter(status__in=['published', 'pending'])
    return render(request, 'admin_courses.html', context)

@login_required
def admin_approve_course(request, id):
    course = get_object_or_404(Course, id=id)
    course.status = 'published'
    course.save()
    return redirect('admin_courses')

@login_required
def admin_reject_course(request, id):
    course = get_object_or_404(Course, id=id)
    course.status = 'rejected'
    course.save()
    return redirect('admin_courses')

@login_required
def admin_list_product(request):
    context = {}
    context['products'] = Product.objects.all()
    return render(request, 'admin_list_product.html', context)

@login_required
def admin_create_product(request):
    if(request.method == 'GET'):
        context = {}
        context['form'] = ProductCreateForm()
        return render(request, 'admin_create_product.html', context)

    elif(request.method == 'POST'):
        form = ProductCreateForm(request.POST, request.FILES)

        if form.is_valid():
            product = form.save()
            return redirect(reverse('admin_list_product'))

        context['form'] = form
        return render(request, 'admin_upsert_product.html', context)


@login_required
def admin_edit_product(request, id):
    product = get_object_or_404(Product, id=id)
    context = {}

    if request.method == 'GET':
        context['form'] = ProductCreateForm(instance=product)
        return render(request, 'admin_create_product.html', context)

    elif request.method == 'POST':
        form = ProductCreateForm(data=request.POST, files=request.FILES, instance=product)

        if form.is_valid():
            product = form.save()
            return redirect('admin_list_product')

        context['form'] = form
        return render(request, 'admin_create_product.html', context)


@login_required
def admin_delete_product(request, id):
    product = get_object_or_404(Product, id=id)
    product.status = 'inactive'
    product.save()
    return redirect(reverse('admin_list_product'))

class EventListView(ListView):
    template_name = 'admin_list_event.html'
    model = Event
    template_name = 'admin_list_event.html'

class EventCreateView(LoginRequiredMixin, CreateView):
    template_name = 'admin_create_event.html'
    model = Event
    form_class = CreateEventForm
    success_url = reverse_lazy('admin_event')

class EventUpdateView(LoginRequiredMixin, UpdateView):
    template_name = 'admin_create_event.html'
    model = Event
    form_class = CreateEventForm
    success_url = reverse_lazy('admin_event')
    pk_url_kwarg = 'id'

@login_required
def admin_delete_event(request, id):
    product = get_object_or_404(Event, id=id)
    product.delete()
    return redirect(reverse('admin_event'))