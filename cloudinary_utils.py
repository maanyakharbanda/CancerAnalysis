import os
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader

load_dotenv()
# Configure Cloudinary using environment variables
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
    secure=True
)

def upload_image_to_cloudinary(image, public_id=None, folder="results"):
    """
    Uploads a PIL Image to Cloudinary and returns the secure_url.
    """
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    upload_preset = os.environ.get("CLOUDINARY_UPLOAD_PRESET")
    upload_result = cloudinary.uploader.upload(
        buf,
        folder=folder,
        public_id=public_id,
        overwrite=True,
        resource_type="image",
        upload_preset=upload_preset
    )
    return upload_result.get("secure_url")