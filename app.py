import streamlit as st
import numpy as np
import tempfile
import os
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io

# Page configuration
st.set_page_config(
    page_title="Pine Flower Phenology Recognition",
    page_icon="🌲",
    layout="wide"
)

# Model file check
model_path = 'models/best.pt'
if os.path.exists(model_path):
    st.sidebar.success(f"✅ Model file loaded successfully ({os.path.getsize(model_path)/1024/1024:.1f} MB)")
else:
    st.sidebar.error("❌ Model file not found")

# Pine flower phenology classes mapping
PINE_FLOWER_CLASSES = {
    0: {'name': 'elongation stage', 'color': (0, 255, 0), 'display_name': 'Elongation Stage'},
    1: {'name': 'ripening stage', 'color': (255, 165, 0), 'display_name': 'Ripening Stage'},
    2: {'name': 'decline stage', 'color': (255, 0, 0), 'display_name': 'Decline Stage'}
}


class StreamlitDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """Load PMC_PhaseNet model"""
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            self.model = None

    def detect_image(self, image):
        """Perform image detection"""
        try:
            st.write("---")
            st.write("🔍 **Starting detection process**")

            if self.model is not None:
                st.write("✅ Using PMC_PhaseNet model for detection...")

                # Perform detection - YOLO can handle PIL Images directly
                results = self.model(image)
                st.write(f"📊 PMC_PhaseNet returned {len(results)} detection results")

                detections = []
                for i, result in enumerate(results):
                    boxes = result.boxes
                    if boxes is not None:
                        st.write(f"🎯 Result {i + 1}: Detected {len(boxes)} targets")

                        for j, box in enumerate(boxes):
                            class_id = int(box.cls.item())
                            confidence = box.conf.item()
                            bbox = box.xyxy[0].tolist()

                            st.write(f"   📦 Target {j + 1}:")
                            st.write(
                                f"     Class: {class_id} ({PINE_FLOWER_CLASSES.get(class_id, {}).get('display_name', 'Unknown')})")
                            st.write(f"     Confidence: {confidence:.3f}")
                            st.write(f"     Location: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")

                            class_info = PINE_FLOWER_CLASSES.get(class_id, {
                                'name': 'unknown', 'color': (255, 165, 0), 'display_name': 'Unknown Stage'
                            })

                            detections.append({
                                'bbox': bbox,
                                'confidence': confidence,
                                'class_name': class_info['name'],
                                'display_name': class_info['display_name'],
                                'class_id': class_id,
                                'color': class_info['color']
                            })
                    else:
                        st.warning(f"⚠️ Result {i + 1}: No targets detected")

                st.write(f"🎉 Total detected: {len(detections)} pine flowers")

            else:
                st.warning("⚠️ Model not loaded, using simulated detection")
                detections = self.mock_detect(image)

            # Draw detection results
            st.write("🖌️ Starting to draw detection boxes...")
            result_image = self.draw_detections(image, detections)
            return detections, result_image

        except Exception as e:
            st.error(f"❌ Error during detection: {e}")
            import traceback
            st.error("Error details:")
            st.code(traceback.format_exc())
            return self.mock_detect(image), image

    def draw_detections(self, image, detections):
        """Draw detection boxes on image using PIL"""
        st.write(f"🖌️ Need to draw {len(detections)} detection boxes")

        if len(detections) == 0:
            st.warning("⚠️ No detection boxes to draw, returning original image")
            return image

        # Create a copy of the image to draw on
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image.astype('uint8'))
        else:
            pil_image = image.copy()

        draw = ImageDraw.Draw(pil_image)
        
        # Use default font
        try:
            # Try to load a font
            font = ImageFont.truetype("Arial.ttf", 16)
        except:
            # Use default font if Arial is not available
            font = ImageFont.load_default()

        image_width, image_height = pil_image.size
        st.write(f"📏 Canvas dimensions: width={image_width}, height={image_height}")

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            color = det.get('color', (0, 255, 0))
            display_name = det['display_name']

            st.write(f"  🎨 Drawing box {i + 1}: {display_name}")
            st.write(f"     Confidence: {conf:.2f}")
            st.write(f"     Coordinates: [{x1}, {y1}, {x2}, {y2}]")

            # Check if coordinates are valid
            if x1 >= x2 or y1 >= y2:
                st.error(f"     ❌ Invalid coordinates: x1>=x2 or y1>=y2")
                continue

            if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
                st.warning(f"     ⚠️ Coordinates partially outside image boundaries")

            # Draw detection box (thicker border)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            st.write(f"     ✅ Bounding box drawn")

            # Draw label
            label = f"{display_name} {conf:.2f}"
            
            # Estimate text size
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                # Fallback if textbbox fails
                text_width = len(label) * 8
                text_height = 16

            # Calculate label position
            label_y = max(y1 - text_height - 5, 5)
            
            # Draw label background
            draw.rectangle([x1, label_y, x1 + text_width + 10, label_y + text_height + 5], 
                         fill=color)
            
            # Draw label text
            draw.text((x1 + 5, label_y + 2), label, fill=(255, 255, 255), font=font)
            st.write(f"     ✅ Label drawn")

        st.success("🎨 All detection boxes drawn successfully!")
        return pil_image

    def mock_detect(self, image):
        """Simulated detection for testing"""
        if isinstance(image, np.ndarray):
            height, width = image.shape[:2]
        else:
            width, height = image.size
            
        detections = []
        import random
        num_detections = random.randint(2, 4)

        for i in range(num_detections):
            x1 = random.randint(50, width - 150)
            y1 = random.randint(50, height - 150)
            x2 = x1 + random.randint(80, 200)
            y2 = y1 + random.randint(80, 200)
            confidence = round(0.7 + random.random() * 0.25, 2)
            class_id = random.randint(0, 2)
            class_info = PINE_FLOWER_CLASSES[class_id]

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_name': class_info['name'],
                'display_name': class_info['display_name'],
                'class_id': class_id,
                'color': class_info['color']
            })
        return detections

    def get_statistics(self, detections):
        """Get detection statistics"""
        stats = {'total_count': 0, 'by_stage': defaultdict(int)}
        if not detections:
            return stats

        stats['total_count'] = len(detections)
        for det in detections:
            stage = det['display_name']
            stats['by_stage'][stage] += 1

        return stats


# Initialize detector
@st.cache_resource
def load_detector():
    return StreamlitDetector('models/best.pt')


def main():
    # Title
    st.title("🌲 Pine Flower Phenology Recognition System")
    st.markdown("Based on PMC_PhaseNet - Detect elongation, ripening, and decline stages")

    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info("This system uses PMC_PhaseNet to detect and classify pine flower phenology stages.")

    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Supported formats: JPG, PNG, JPEG"
    )

    if uploaded_file is not None:
        # Display file information
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024 / 1024:.2f} MB",
            "File type": uploaded_file.type
        }
        st.write("File details:", file_details)

        # Load detector
        detector = load_detector()

        if st.button("Start Detection", type="primary"):
            with st.spinner("Processing..."):
                try:
                    # Load image using PIL
                    image = Image.open(uploaded_file).convert('RGB')
                    
                    # Display original image
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Image")
                        st.image(image, use_container_width=True)

                    # Detection
                    detections, result_image = detector.detect_image(image)

                    # Display results
                    with col2:
                        st.subheader("Detection Result")
                        st.image(result_image, use_container_width=True)

                    # Display statistics
                    st.subheader("📊 Detection Statistics")
                    stats = detector.get_statistics(detections)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Detections", stats['total_count'])

                    with col2:
                        for stage, count in stats['by_stage'].items():
                            st.metric(f"{stage}", count)

                    # Display detection details
                    st.subheader("🔍 Detection Details")
                    if detections:
                        for i, det in enumerate(detections):
                            st.write(
                                f"**Pine Flower {i + 1}**: {det['display_name']} (Confidence: {det['confidence']:.2f})")
                    else:
                        st.info("No pine flowers detected")

                    st.success(f"Detection completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    import traceback
                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
