import uuid
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login as auth_login, authenticate, logout as auth_logout
from django.contrib import messages
from django.core.mail import EmailMultiAlternatives
from django.conf import settings
from django.contrib.auth.decorators import login_required
from ..forms import *
from django.urls import reverse
from django.http import HttpResponse
from ..models import *
from django.http import HttpResponseBadRequest
from django.contrib.auth import get_user_model

def index(request):
    return render(request, 'index.html')

def courses(request):
    return render(request, 'courses.html')

def category(request):
    return render(request, 'courses.html')

def events(request):
    return render(request, 'events.html')

def contact(request):
    return render(request, 'contact.html')

def logout_view(request):
    logout(request)
    return redirect('login') 



def student(request):
    return render(request, 'student.html')

def teacher(request):
    return render(request, 'teacher.html')

def teacher_qualification(request):
    return render(request, 'teacher_qualification.html')

def view_categories(request):
    return render(request, 'add_category.html')

def add_product_rent(request):
    return render(request, 'add_product_rent.html')

def add_product_buy(request):
    return render(request, 'add_product_buy.html')

def admin_index(request):
    return render(request, 'adminindex.html')

def add_subcategory(request):
    # Your view logic here
    return render(request, 'create_subcategory.html')

def student_details(request):
    # Your logic here
    return render(request, 'student_details.html')

User = get_user_model()

@login_required
def edit_profile(request):
    user = request.user
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')

        user.username = username
        user.email = email
        user.save()
        messages.success(request, 'Your profile was successfully updated!')
        return redirect('profile')  # Adjust this redirect as needed

    return render(request, 'edit_profile.html', {'user': user})

@login_required
def delete_subcategory(request, subcategory_id):
    if request.method == 'POST':
        subcategory = get_object_or_404(SubCategory, id=subcategory_id)
        subcategory.delete()
    return redirect('view_subcategories')

@login_required
def profile(request):
    user = request.user
    context = {
        'user': user,
    }
    return render(request, 'profile.html', context)

@login_required
def profile_view(request):
    student = request.user  # Assuming user is a student
    return render(request, 'profile.html', {'student': student})

@login_required
def view_subcategories(request):
    subcategories = SubCategory.objects.all()  # Or use filters if needed
    return render(request, 'view_subcategories.html', {'subcategories': subcategories})

@login_required
def update_request_status(request, request_id, status):
    if not request.user.is_superuser:
        return redirect('home')  # Ensure only admins can update status

    teacher_request = get_object_or_404(TeacherRequest, pk=request_id)
    if status in ['Accepted', 'Rejected']:
        teacher_request.status = status
        teacher_request.save()
    return redirect(reverse('admin:app_teacherrequest_changelist'))  # Redirect back to admin list view

def submit_teacher_request(request):
    if request.method == 'POST':
        form = TeacherRequestForm(request.POST, request.FILES)
        if form.is_valid():
            teacher_request = form.save(commit=False)
            teacher_request.user = request.user
            teacher_request.save()
            return redirect('approval_pending')  # Redirect to an approval pending page
    else:
        form = TeacherRequestForm()
    return render(request, 'teacher_qualification.html', {'form': form})

@login_required
def approval_pending(request):
    return render(request, 'approval_pending.html')

@login_required
def teacher_requests(request):
    requests = TeacherRequest.objects.filter(status='pending')
    return render(request, 'admin_teacher_requests.html', {'requests': requests})

@login_required
def teacher_requests(request):
    if request.method == 'POST':
        action = request.POST.get('action')
        request_id = request.POST.get('request_id')
        teacher_request = TeacherRequest.objects.get(id=request_id)

        if action == 'accept':
            teacher_request.approved = True
        elif action == 'reject':
            teacher_request.rejected = True

        teacher_request.save()
        return redirect('teacher_requests')

    teacher_requests = TeacherRequest.objects.filter(approved=False, rejected=False)
    return render(request, 'admin_teacher_requests.html', {'teacher_requests': teacher_requests})

@login_required
def approve_request(request, request_id):
    teacher_request = get_object_or_404(TeacherRequest, id=request_id)
    teacher_request.status = 'approved'
    teacher_request.save()
    return redirect('teacher_requests')

@login_required
def reject_request(request, request_id):
    teacher_request = get_object_or_404(TeacherRequest, id=request_id)
    teacher_request.status = 'rejected'
    teacher_request.save()
    return redirect('teacher_requests')

@login_required
def accept_teacher(request):
    if request.method == 'POST':
        request_id = request.POST.get('request_id')
        teacher_request = TeacherRequest.objects.get(id=request_id)
        teacher_request.status = 'Accepted'
        teacher_request.save()
        messages.success(request, 'Teacher request accepted.')
    return redirect('teacher_requests')

@login_required
def reject_teacher(request):
    if request.method == 'POST':
        request_id = request.POST.get('request_id')
        teacher_request = TeacherRequest.objects.get(id=request_id)
        teacher_request.status = 'Rejected'
        teacher_request.save()
        messages.error(request, 'Teacher request rejected.')
    return redirect('teacher_requests')

def password_reset(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        try:
            user = User.objects.get(email=email)
            token = str(uuid.uuid4())
            msg = f'http://127.0.0.1:8000/reset/{token}/'
            email = EmailMultiAlternatives('Password Reset', msg, settings.EMAIL_HOST_USER, [user.email])
            email.send()
            PasswordReset.objects.create(user_id=user.id, token=token)
            return render(request, 'password_reset.html', {'message': 'Password reset link has been sent to your mail. Kindly check your mail and follow the procedures.'})
        except User.DoesNotExist:
            return render(request, 'password_reset.html', {'error': 'User with this email does not exist'})
    return render(request, 'password_reset.html')

def reset(request, token):
    reset_user = PasswordReset.objects.filter(token=token).first()
    if reset_user is None:
        return render(request, 'reset.html', {'message': 'Invalid or expired token'})
    uid = reset_user.user_id
    if request.method == 'POST':
        new_password = request.POST['new_password']
        confirm_password = request.POST['confirm_password']
        user_id = request.POST.get('user_id')
        if user_id is None:
            return render(request, 'reset.html', {'message': 'No user found'})
        if new_password != confirm_password:
            return render(request, 'reset.html', {'error': 'Passwords do not match'})
        user_obj = User.objects.get(id=user_id)
        user_obj.set_password(new_password)
        user_obj.save()
        return redirect('login')
    return render(request, 'reset.html', {'user_id': uid})

def user_login(request):
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']
        user = authenticate(request, email=email, password=password)
        if user is not None:
            auth_login(request, user)
            if user.usertype == 'admin':
                return redirect('admin_view')
            elif user.usertype == 'student':
                return redirect(reverse('student_home'))
            elif user.usertype == 'teacher':
                #check if the profile is approved
                if TeacherRequest.objects.filter(user=user, status='Accepted').exists():
                    return redirect('teacher_home')
                else:
                    return redirect('teacher_qualification')
        else:
            messages.error(request, 'Invalid email or password')
    return render(request, 'login.html')

def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        email = request.POST['email']
        usertype = request.POST['usertype']
        if not username or not password or not email or not usertype:
            messages.error(request, 'All fields are required')
            return render(request, 'register.html')
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already taken')
            return render(request, 'register.html')
        if User.objects.filter(email=email).exists():
            messages.error(request, 'Email already registered')
            return render(request, 'register.html')
        user = User.objects.create_user(username=username, password=password, email=email, usertype=usertype)
        user.save()
        messages.success(request, 'Registration successful. You are now logged in.')
        return redirect('login')
    return render(request, 'register.html')

def logout(request):
    auth_logout(request)
    return redirect('login')

def teacher_qualification(request):
    teacher_request = TeacherRequest.objects.filter(user=request.user).first()

    if request.method == 'POST':
        full_name = request.POST.get('full_name')
        email = request.POST.get('email')
        phone_number = request.POST.get('phone_number')
        qualification = request.POST.get('qualification')
        experience_years = request.POST.get('experience_years')
        skills = request.POST.get('skills')
        resume = request.FILES.get('resume')

        # Check if all fields are filled
        if not (full_name and email and phone_number and qualification and experience_years and skills and resume):
            return render(request, 'teacher.html', {
                'error_message': 'All fields are required.',
                'teacher_request': teacher_request
            })

        if teacher_request:
            # Update existing request
            teacher_request.full_name = full_name
            teacher_request.email = email
            teacher_request.phone_number = phone_number
            teacher_request.qualification = qualification
            teacher_request.experience_years = int(experience_years)
            teacher_request.skills = skills
            teacher_request.resume = resume
            teacher_request.save()
        else:
            # Create a new request
            TeacherRequest.objects.create(
                user=request.user,
                full_name=full_name,
                email=email,
                phone_number=phone_number,
                qualification=qualification,
                experience_years=int(experience_years),
                skills=skills,
                resume=resume,
            )

        # Redirect based on the status
        if teacher_request and teacher_request.status == 'Accepted':
            return redirect('teacher_in')
        elif teacher_request and teacher_request.status == 'Rejected':
            return redirect('home')
        else:
            return redirect('teacher_qualification')
    elif request.method == 'GET':
        print(teacher_request)
        return render(request, 'teacher.html', {
            'teacher_request': teacher_request
        })
    
@login_required
def admin_teacher_requests_view(request):
    if request.method == 'GET':
        teacher_requests = TeacherRequest.objects.all()
        return render(request, 'admin_teacher_requests.html', {'teacher_requests': teacher_requests})
    elif request.method == 'POST':
        request_id = request.POST.get('request_id')
        action = request.POST.get('action')
        teacher_request = get_object_or_404(TeacherRequest, id=request_id)
        if action == 'accept':
            teacher_request.approved = True
            teacher_request.rejected = False
        elif action == 'reject':
            teacher_request.approved = False
            teacher_request.rejected = True
        teacher_request.save()
        return redirect('teacher_requests')
    teacher_requests = TeacherRequest.objects.filter(approved=False, rejected=False)
    return render(request, 'admin_teacher_requests.html', {'teacher_requests': teacher_requests})

# @login_required
# def teacher_requests_view(request):
#     teacher_requests = TeacherRequest.objects.all()
#     return render(request, 'admin_teacher_requests.html', {'teacher_requests': teacher_requests})

@login_required
def add_category(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description')
        if name and description:
            Category.objects.create(name=name, description=description)
            return redirect('view_categories')  # Redirect to categories list after saving
        else:
            return HttpResponse("All fields are required", status=400)
    return render(request, 'add_category.html')

@login_required
def view_categories(request):
    categories = Category.objects.all()
    return render(request, 'view_categories.html', {'categories': categories})

@login_required
def edit_category(request, id):
    category = get_object_or_404(Category, id=id)
    
    if request.method == 'POST':
        category.name = request.POST.get('name', category.name)
        category.description = request.POST.get('description', category.description)
        category.save()
        return redirect('view_categories')
    
    return render(request, 'edit_category.html', {'category': category})

@login_required
def delete_category(request, id):
    category = get_object_or_404(Category, id=id)
    if request.method == 'POST':
        category.delete()
        return redirect('view_categories')
    return redirect('view_categories')

@login_required
def create_subcategory(request):
    if request.method == 'POST':
        category_id = request.POST.get('category')
        subcategory_name = request.POST.get('name')

        if not category_id or not subcategory_name:
            return HttpResponseBadRequest('Invalid input')

        try:
            category = Category.objects.get(id=category_id)
        except Category.DoesNotExist:
            return HttpResponseBadRequest('Category does not exist')

        SubCategory.objects.create(name=subcategory_name, category=category)
        return redirect('subcategory_list')  # Ensure 'subcategory_list' is a valid URL pattern name

    else:
        categories = Category.objects.all()
        return render(request, 'create_subcategory.html', {'categories': categories})

@login_required
def add_subcategory(request):
    if request.method == 'POST':
        category_id = request.POST.get('category')
        name = request.POST.get('name')
        image = request.FILES.get('image')  # Get the uploaded image
        
        category = Category.objects.get(id=category_id)
        
        # Create a new SubCategory instance with the image
        subcategory = SubCategory(name=name, category=category, image=image)
        subcategory.save()
        
        return redirect('view_subcategories')  # Redirect to a list of subcategories or wherever suitable
    
    categories = Category.objects.all()
    return render(request, 'add_subcategory.html', {'categories': categories})


def contact_us(request):
    if request.method == 'POST':
        form = ContactUsForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('contact')
        else:
            return render(request, 'contact_us.html', {'form': form})

    form = ContactUsForm()
    return render(request, 'contact_us.html', {'form': form})



