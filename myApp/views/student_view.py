from django.shortcuts import render, redirect, get_object_or_404, HttpResponse
from django.urls import reverse
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import ListView, DetailView, TemplateView
import stripe
from django.conf import settings
from ..models import *
from ..forms import *
from ..models import Feedback  # Assuming you have a Feedback model
from django.contrib import messages
from django.contrib.messages import success
from ..ml_model.ml_model import predict_art_form
import os
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
from django.conf import settings
import fitz  # PyMuPDF
import os
from datetime import datetime
import uuid

def find_art(request):
    if request.method == 'POST' and request.FILES['art_image']:
        art_image = request.FILES['art_image']
        
        # Create a path for the temporary file
        temp_path = os.path.join('temp', art_image.name)
        
        # Save the uploaded file
        path = default_storage.save(temp_path, ContentFile(art_image.read()))
        
        # Get the full path of the saved file
        full_path = os.path.join(settings.MEDIA_ROOT, path)
        
        print("Full path:", full_path)  # Debug print
        
        # Predict the art form
        art_form, description = predict_art_form(full_path)
        
        # Delete the temporary file
        default_storage.delete(path)

        # Render the results in the template
        context = {
            'art_form': art_form,
            'description': description
        }
        return render(request, 'find_art.html', context)

    return render(request, 'find_art.html')

def submit_feedback(request, lesson_id):
    if request.method == 'POST':
        rating = request.POST['rating']
        feedback_text = request.POST['feedback']
        user = request.user
        lesson = CourseLesson.objects.get(id=lesson_id)

        # Save the feedback to the database
        Feedback.objects.create(user=user, lesson=lesson, rating=rating, feedback=feedback_text)

        # Use messages to show a success message
        messages.success(request, 'Thank you for your feedback!')
        return redirect('student_view_lesson', id=lesson_id)

    
@login_required
def student_home(request):
    context = {}
    context['courses'] = EnrolledCourse.objects.prefetch_related('course__lessons').filter(student=request.user)
    q = request.GET.get('q')
    if q:
        context['courses'] = context['courses'].filter(course__name__icontains=q)
    return render(request, 'student_home.html', context)

def student_more_courses(request):
    context = {}
    enrolled_course_ids = list(EnrolledCourse.objects.filter(student=request.user).values_list('course_id', flat=True))
    context['courses'] = Course.objects.filter(status='published').exclude(id__in=enrolled_course_ids)
    q = request.GET.get('q')
    if q:
        context['courses'] = context['courses'].filter(name__icontains=q)
    return render(request, 'student_more_courses.html', context)

def student_enroll(request, id):
    course = get_object_or_404(Course, id=id)
    # Check if the user has already enrolled in the course.
    if EnrolledCourse.objects.filter(course=course, student=request.user).exists():
        return redirect(reverse('student_home'))
    
    context = {
        'stripe_public_key': settings.STRIPE_PUBLIC_KEY,
        'course': course
    }
    
    if request.method == 'GET':
        return render(request, 'student_enroll.html', context)
    # EnrolledCourse.objects.create(course=course, student=request.user)
    return redirect(reverse('student_home'))

def student_enroll_success(request, id):
    course = get_object_or_404(Course, id=id)
    # Check if the user has already enrolled in the course.
    if EnrolledCourse.objects.filter(course=course, student=request.user).exists():
        return redirect(reverse('student_home'))
    
    EnrolledCourse.objects.create(course=course, student=request.user)
    return redirect(reverse('student_home'))

def student_enroll_failure(request):
    return redirect(reverse('student_home'))

@csrf_exempt
def student_create_course_payment(request, id):
    stripe.api_key = settings.STRIPE_SECRET_KEY
    course = get_object_or_404(Course, id=id)
    if request.method == 'POST':
        try:
            # Create a new Checkout Session for the payment
            checkout_session = stripe.checkout.Session.create(
                payment_method_types=['card'],  # Payment methods allowed
                line_items=[{
                    'price_data': {
                        'currency': 'inr',
                        'product_data': {
                            'name': course.name,
                        },
                        'unit_amount': int(course.price * 100),
                    },
                    'quantity': 1,
                }],
                mode='payment',
                success_url=f'http://127.0.0.1:8000/student/enroll/success/{course.id}',  # Redirect after success
                cancel_url='http://127.0.0.1:8000/student/enroll/failre/',     # Redirect if canceled
            )
            # Redirect the user to the Stripe Checkout page
            return JsonResponse({'id': checkout_session.id})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=403)

def student_take_lesson(request, id):
    lesson = get_object_or_404(CourseLesson, id=id)
    course = Course.objects.filter(id=lesson.course_id)
    lessons = CourseLesson.objects.filter(course_id=lesson.course_id)
    context = {
        'course': course,
        'lesson': lesson,
        'lessons': lessons
    }
    return render(request, 'student_view_lesson.html', context)

def student_subtitle_lesson(request, id, language):
    print(id, language)
    subtitle = get_object_or_404(LessonVideoSubtitle, lesson_id=id, language=language)
    print(subtitle.subtitle)
    return HttpResponse(subtitle.subtitle, content_type='text/plain')

def student_store(request):
    products = Product.objects.filter(status='active')
    return render(request, 'student_store.html', {'products': products})

def student_cart_add(request, id):
    product = get_object_or_404(Product, id=id)
    # Add to cart if the product is not added yet
    if Cart.objects.filter(product=product, user=request.user).exists():
        cart = Cart.objects.get(product=product, user=request.user)
        cart.quantity += 1
        cart.save()
    else:
        Cart.objects.create(product=product, user=request.user, quantity=1)
    return redirect(reverse('student_cart'))

def student_cart_delete(request, id):
    cart = get_object_or_404(Cart, id=id)
    cart.delete()
    return redirect(reverse('student_cart'))

def student_cart(request):
    context = {}
    context['carts'] = Cart.objects.filter(user=request.user)
    return render(request, 'student_cart.html', context)

def student_place_order(request):
    context = {}
    if request.method == 'GET':
        context['form'] = OrderForm()
        return render(request, 'student_place_order.html', context)

    elif request.method == 'POST':
        form = OrderForm(data=request.POST)
        context['form'] = form
        if form.is_valid():
            order = form.save(commit=False)
            order.user = request.user

            # Calculating total price
            total = 0
            for cart in Cart.objects.filter(user=request.user):
                total += cart.quantity * cart.product.price

            order.price = total
            order.save()
            for cart in Cart.objects.filter(user=request.user):
                OrderItem.objects.create(
                    order=order,
                    product=cart.product,
                    quantity=cart.quantity
                )
            # Redirect to payment page and pass order id
            return redirect(reverse('student_order_payment', args=[order.id]))
        return render(request, 'student_place_order.html', context)

def student_order_payment(request, id):
    context = {}
    context['stripe_public_key'] = settings.STRIPE_PUBLIC_KEY
    context['order'] = get_object_or_404(Order, id=id)
    return render(request, 'student_order_payment.html', context)

def student_order_success(request, id):
    order = get_object_or_404(Order, id=id)
    order.success = True
    # clear cart items
    Cart.objects.filter(user=request.user).delete()
    order.save()
    return redirect(reverse('student_orders'))

def student_orders(request):
    context = {}
    context['orders'] = Order.objects.filter(user=request.user, success=True)
    return render(request, 'student_orders.html', context)

@csrf_exempt
def student_create_order_payment(request, id):
    stripe.api_key = settings.STRIPE_SECRET_KEY
    order = get_object_or_404(Order, id=id)
    order_items = OrderItem.objects.filter(order=order)
    if request.method == 'POST':
        try:
            # Create a new Checkout Session for the payment
            checkout_session = stripe.checkout.Session.create(
                payment_method_types=['card'],  # Payment methods allowed
                # Create order items based on order

                line_items=[{
                    'price_data': {
                        'currency': 'inr',
                        'product_data': {
                            'name': 'Store Order',
                        },
                        'unit_amount': int(order.price * 100),
                    },
                    'quantity': 1,
                }],
                mode='payment',
                success_url=f'http://127.0.0.1:8000/student/orders/success/{order.id}',  # Redirect after success
                cancel_url='http://127.0.0.1:8000/student/cart/',     # Redirect if canceled
            )
            # Redirect the user to the Stripe Checkout page
            return JsonResponse({'id': checkout_session.id})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=403)

class StudentEventListView(LoginRequiredMixin, ListView):
    model = Event
    template_name = 'student_events.html'
    context_object_name = 'events'

    def get_queryset(self):
        queryset = super(StudentEventListView, self).get_queryset()
        # Distinct categories
        categories = EnrolledCourse.objects.filter(student=self.request.user).values_list('course__sub_category__category', flat=True)
        categories = list(set(categories))
        queryset = queryset.filter(category_id__in=categories)
        return queryset

class StudentEventDetailView(LoginRequiredMixin, DetailView):
    model = Event
    template_name = 'student_event_detail.html'
    context_object_name = 'event'
    pk_url_kwarg = 'id'

class StudentLiveClassListView(LoginRequiredMixin, ListView):
    model = LiveClass
    template_name = 'student_list_liveclass.html'

    def get_queryset(self):
        enrolled_courses = EnrolledCourse.objects.filter(student=self.request.user)
        course_ids = enrolled_courses.values_list('course_id', flat=True)
        queryset = super(StudentLiveClassListView, self).get_queryset().filter(course_id__in=course_ids)
        return queryset

class StudentNotificationListView(LoginRequiredMixin, ListView):
    model = Notification
    template_name = 'student_list_notification.html'
    
    def get_queryset(self):
        return super().get_queryset().filter(user=self.request.user)

class GetCertificateView(LoginRequiredMixin, TemplateView):
    template_name = 'student_get_certificate.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['course'] = EnrolledCourse.objects.prefetch_related('course').prefetch_related('course__instructor').filter(student=self.request.user).first()
        course = context['course'].course
        # Get the certificate details if exists. Else create one.
        certificate = None
        if CourseCertificate.objects.filter(user=self.request.user, course=course).exists():
            certificate = CourseCertificate.objects.filter(user=self.request.user).first()
        else:
            static_dir = os.path.join(settings.STATICFILES_DIRS[0], 'certificates')
            # File paths
            input_pdf = "certificate.pdf"
            # output_pdf = "output.pdf"
            output_filename = f'{uuid.uuid4()}.pdf'
            output_pdf = os.path.join(static_dir, output_filename)
            # # Get the current date
            current_date = datetime.now()

            # # Format the date in dd/MM/yyyy format
            formatted_date = current_date.strftime(f"\n%d/%m/%Y")
            print(self.request.user)
            
            # # Dictionary of multiple replacements
            replacements = {
                "STUDENT": f'{self.request.user.first_name} {self.request.user.last_name}',
                "INSTRUCTORNAME": f'{context['course'].course.instructor.first_name} {context['course'].course.instructor.last_name}',
                'COURSE': f'{context["course"].course.name}',
                "DATE": formatted_date
            }
            
            # # Verify input file exists
            if not os.path.exists(input_pdf):
                print(f"Error: Input file '{input_pdf}' not found")
                return context
            
            replace_multiple_in_pdf(input_pdf, output_pdf, replacements)

            certificate = CourseCertificate(user=self.request.user, course=course, certificate_url=output_filename)
            certificate.save()
            context['certificate'] = certificate
        
        context['certificate_url'] = os.path.join(settings.STATIC_URL, 'certificates', certificate.certificate_url)
        return context

# Helper functions for PDF Certificate Generation.
def get_font_info(page, text_instance):
    """Extract font details from a text instance."""
    text_dict = page.get_text("dict")
    for block in text_dict["blocks"]:
        if block["type"] == 0:  # Text block
            for line in block["lines"]:
                for span in line["spans"]:
                    span_rect = fitz.Rect(span["bbox"])
                    if span_rect.intersects(text_instance):
                        # Convert color from integer to RGB tuple
                        color_int = span["color"]
                        r = (color_int >> 16) & 255
                        g = (color_int >> 8) & 255
                        b = color_int & 255
                        color_rgb = (r / 255, g / 255, b / 255)
                        return {
                            "fontname": span["font"],
                            "fontsize": span["size"],
                            "color": color_rgb
                        }
    # Default fallback
    return {"fontname": "helv", "fontsize": 11, "color": (0, 0, 0)}

def replace_multiple_in_pdf(input_pdf, output_pdf, replacements, font_file=None):
    try:
        # Open the input PDF
        pdf_document = fitz.open(input_pdf)
        
        # Load external font if provided
        if font_file and os.path.exists(font_file):
            pdf_document.insert_font(fontname="customfont", fontfile=font_file)
            fallback_font = "customfont"
        else:
            fallback_font = "helv"  # Default Helvetica
        
        # Iterate through each page
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Process each replacement
            for old_text, new_text in replacements.items():
                text_instances = page.search_for(old_text)
                
                if text_instances:
                    print(f"Page {page_num + 1}: Found '{old_text}' at {len(text_instances)} locations")
                    
                    for inst in text_instances:
                        # Get font information
                        font_info = get_font_info(page, inst)
                        
                        # Use original font if available, otherwise fallback
                        try_font = font_info["fontname"]
                        try:
                            # Test if the font is usable
                            page.insert_text((0, 0), "test", fontname=try_font, fontsize=1)
                            page.delete_text((0, 0, 10, 10))  # Clean up test
                        except Exception:
                            try_font = fallback_font
                            print(f"  Font '{font_info['fontname']}' not available, using '{try_font}'")
                        
                        # Erase original text
                        page.draw_rect(inst, color=(1, 1, 1), fill=(1, 1, 1))
                        
                        # Insert new text
                        page.insert_text(
                            (inst.x0, inst.y0),
                            new_text,
                            fontname=try_font,
                            fontsize=font_info["fontsize"],
                            color=font_info["color"]
                        )
                        print(f"  Replaced '{old_text}' with '{new_text}' using {font_info}")
        
        # Save the modified PDF
        pdf_document.save(output_pdf)
        pdf_document.close()
        print(f"Modified PDF saved as: {output_pdf}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")