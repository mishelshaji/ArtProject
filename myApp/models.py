from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db import models
from django.contrib.auth.models import User
from django.conf import settings

class UserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        return self.create_user(email, password, **extra_fields)

class User(AbstractUser):
    USER_TYPE_CHOICES = (
        ('student', 'Student'),
        ('teacher', 'Teacher'),
    )
    email = models.EmailField(max_length=255, unique=True)
    username = models.CharField(max_length=255, unique=True)
    usertype = models.CharField(max_length=10, choices=USER_TYPE_CHOICES)

    objects = UserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

class PasswordReset(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    token = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

class TeacherRequest(models.Model):
    STATUS_CHOICES = [
        ('Pending', 'Pending'),
        ('Accepted', 'Accepted'),
        ('Rejected', 'Rejected'),
    ]
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    full_name = models.CharField(max_length=100)
    email = models.EmailField()
    phone_number = models.CharField(max_length=20)
    qualification = models.CharField(max_length=100)
    experience_years = models.PositiveIntegerField()
    skills = models.CharField(max_length=255)
    resume = models.FileField(upload_to='resumes/')
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='Pending')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.full_name
    
class Category(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField()

    def __str__(self):
        return self.name
    
class SubCategory(models.Model):
    name = models.CharField(max_length=100)
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='subcategories')
    image = models.ImageField(upload_to='subcategories/', blank=True, null=True)
    # other fields as needed

    def __str__(self):
        return self.name 

class ContactUs(models.Model):
    id = models.BigAutoField(primary_key=True)
    first_name = models.CharField(max_length=255)
    last_name = models.CharField(max_length=255)
    email = models.EmailField()
    phone_number = models.CharField(max_length=20, blank=True, null=True)
    message = models.TextField(max_length=1000)

    def __str__(self):
        return self.first_name

class Course(models.Model):
    STATUSES = (
        ('draft', 'Draft'),
        ('published', 'Published'),
        ('archived', 'Archived'),
        ('pending', 'Pending Approval'),
        ('rejected', 'Rejected'),
    )
    name = models.CharField(max_length=255)
    short_description = models.CharField(max_length=255)
    description = models.TextField()
    featured_image = models.ImageField(upload_to='courses/featured_images/')
    sub_category = models.ForeignKey(SubCategory, on_delete=models.CASCADE, related_name='courses')
    duration = models.DurationField()
    instructor = models.ForeignKey(User, on_delete=models.CASCADE, related_name='courses', limit_choices_to={'usertype': 'teacher'})
    learning_outcome = models.TextField(help_text='Place each items on a separate line.')
    price = models.DecimalField(max_digits=8, decimal_places=2)
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=20, choices=STATUSES, default='draft')

    @property
    def is_draft(self):
        return self.status == 'draft'
    
    @property
    def is_published(self):
        return self.status == 'published'
    
    @property
    def is_archived(self):
        return self.status == 'archived'
    
    @property
    def is_pending(self):
        return self.status == 'pending'
    
    @property
    def is_rejected(self):
        return self.status == 'rejected'

    def __str__(self):
        return self.name


class CourseLesson(models.Model):
    id=models.BigAutoField(primary_key=True)
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name='lessons')
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    resource = models.FileField(upload_to='lessons/videos/', blank=True, null=True)
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)
    summary = models.TextField(blank=True, null=True, editable=False)

    def __str__(self):
        return self.title
    
    # A function to return the file type based on the file extension.
    def get_file_type(self):
        if self.resource.name.lower().endswith('.mp4'):
            return 'Video'
        elif self.resource.name.lower().endswith('.pdf'):
            return 'PDF'
        elif self.resource.name.lower().endswith('jpg'):
            return 'Image'
        elif self.resource.name.lower().endswith('png'):
            return 'Image'
        elif self.resource.name.lower().endswith('jpeg'):
            return 'Image'
        else:
            return 'Unknown'


class LessonVideoSubtitle(models.Model):
    LANGUAGE_CHOICES = (
        ('English', 'English'),
        ('Simplified English', 'Simplified English'),
        ('Malayalam', 'Malayalam'),
        ('Tamil', 'Tamil'),
    )
    id = models.BigAutoField(primary_key=True)
    lesson = models.ForeignKey(CourseLesson, on_delete=models.CASCADE)
    subtitle = models.TextField()
    language = models.CharField(max_length=20, choices=LANGUAGE_CHOICES, default='English')
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f'Id: {self.id}, Language: {self.language} Subtitle: {self.subtitle}'


class EnrolledCourse(models.Model):
    id = models.BigAutoField(primary_key=True)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    student = models.ForeignKey(User, on_delete=models.CASCADE)
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)


class Product(models.Model):
    PRODUCT_STATUSES = (
        ('active', 'Active'),
        ('inactive', 'Inactive'),
    )
    name = models.CharField(max_length=255)
    short_description = models.CharField(max_length=255)
    description = models.TextField()
    price = models.DecimalField(max_digits=8, decimal_places=2)
    image = models.ImageField(upload_to='products/')
    stock = models.PositiveIntegerField()
    status = models.CharField(max_length=20, choices=PRODUCT_STATUSES, default='active')
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='products')
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)

    @property
    def stock_status(self):
        if self.stock > 0:
            return 'In Stock'
        else:
            return 'Out of Stock'

    def __str__(self):
        return self.name
    

class Cart(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField()
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)

class Order(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_on = models.DateTimeField(auto_now_add=True)
    price = models.DecimalField(max_digits=8, decimal_places=2)
    address = models.CharField(max_length=255)
    success = models.BooleanField(default=False)

class OrderItem(models.Model):
    id = models.BigAutoField(primary_key=True)
    order = models.ForeignKey(Order, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField()
    created_on = models.DateTimeField(auto_now_add=True)

class LessonReview(models.Model):
    id = models.BigAutoField(primary_key=True)
    lesson = models.ForeignKey(CourseLesson, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    review = models.TextField()
    
class Feedback(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    lesson = models.ForeignKey(CourseLesson, on_delete=models.CASCADE)
    rating = models.IntegerField(choices=[(i, str(i)) for i in range(1, 6)])
    feedback = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

class Event(models.Model):
    id = models.BigAutoField(primary_key=True)
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    category = models.ForeignKey(to=Category, on_delete=models.CASCADE)
    total_seats = models.PositiveIntegerField(default=1)
    banner_image = models.ImageField(upload_to='event_banners/', blank=True, null=True)
    promo_video_url = models.URLField(max_length=250, blank=True, null=True)
    organizer_name = models.CharField(max_length=100, blank=True, null=True)
    registration_url = models.URLField(max_length=250, blank=True, null=True)
    location_url = models.URLField(max_length=250, blank=True, null=True)
    entry_fee = models.DecimalField(max_digits=8, decimal_places=2, default=0.00)
    organizer_phone = models.CharField(max_length=20, blank=True, null=True)
    organizer_email = models.EmailField(max_length=254, blank=True, null=True)
    instructions = models.TextField(max_length=2500, blank=True, null=True)
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)

class LiveClass(models.Model):
    id = models.BigAutoField(primary_key=True)
    title = models.CharField(max_length=255)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    course = models.ForeignKey(to=Course, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    live_url = models.URLField(max_length=250, blank=True, null=True)
    remark = models.TextField(blank=True, null=True)

class Notification(models.Model):
    id = models.BigAutoField(primary_key=True)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)

class CourseCertificate(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    certificate_url = models.URLField(max_length=250, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)