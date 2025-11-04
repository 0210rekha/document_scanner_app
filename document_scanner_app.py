import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="📄 Document Scanner", layout="centered")
st.title("📄 Smart Document Scanner using OpenCV")
st.write("Upload a photo of a document — the app will detect edges, find the paper boundary, and flatten it like a real scanner!")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute new width and height
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Original Image", use_column_width=True)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 75, 200)

    st.image(edges, caption="Step 1: Canny Edge Detection", use_column_width=True, clamp=True)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    doc_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            doc_contour = approx
            break

    if doc_contour is not None:
        output = image.copy()
        cv2.drawContours(output, [doc_contour], -1, (0, 255, 0), 3)
        st.image(output, caption="Step 2: Detected Document Boundary", use_column_width=True)

        # Perspective transform
        warped = four_point_transform(image, doc_contour.reshape(4, 2))
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
        scanned = cv2.adaptiveThreshold(
            warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        st.image(scanned, caption="📄 Step 3: Scanned Document Output", use_column_width=True, clamp=True)
        st.success("✅ Document scanned successfully!")

    else:
        st.error("❌ No document contour detected. Try a clearer photo with visible edges.")
