# Android Mobile & Live Camera Verification Checklist

This checklist provides a systematic step-by-step procedure for verifying the **AI-Powered Soil Analytics System** on physical Android devices (including Android Chrome, Samsung Internet, and tablet browsers).

---

## 1. Network & Protocol Prerequisites

- [ ] **HTTPS / Localhost Requirement**:
  - WebRTC camera access (`navigator.mediaDevices.getUserMedia`) requires a **Secure Context** (HTTPS or `localhost`).
  - For local LAN testing on an Android device:
    - Option 1: Use Chrome remote port forwarding (`chrome://inspect/#devices`) mapping `localhost:8000` to the phone's `localhost:8000`.
    - Option 2: Deploy behind a reverse proxy (e.g., Caddy or Nginx) with an SSL/TLS certificate.
- [ ] **Supported Browsers**:
  - Android Chrome (v110+)
  - Firefox for Android (v115+)
  - Samsung Internet (v20+)

---

## 2. Module 1: Image Input Verification

### Option A: Gallery / File Upload
1. Open the dashboard on Android Chrome.
2. Navigate to the **Analyze Soil** tab.
3. Tap **Option A: Select from Gallery**.
4. Verify that the Android system file/photo picker appears.
5. Select a JPEG or PNG soil photograph from the device gallery.
6. **Preview Verification**:
   - Verify the thumbnail preview is displayed immediately.
   - Verify the filename and file size (in MB) are rendered correctly.
7. Tap **Upload & Preprocess**:
   - Ensure a progress toast or notification appears.
   - Verify the upload succeeds with HTTP 201.
   - Verify the OpenCV side-by-side comparison (Original vs. 224×224 CLAHE preprocessed) renders cleanly.

### Option B: Live Camera Capture
1. In the **Analyze Soil** tab, locate **Option B: Live Camera Capture**.
2. Tap **📷 Open Camera**.
3. **Permission Prompt**:
   - Observe the Android Chrome permission prompt: *"Allow soil-analytics to use your camera?"*.
   - Tap **Allow**.
4. **Camera Stream & Rear Lens**:
   - Confirm the live video stream activates smoothly without distortion or freezing.
   - Verify that the **rear camera** is selected by default (`facingMode: { ideal: "environment" }`).
5. **Capture Flow**:
   - Point the camera at a soil sample in good natural lighting (15–20 cm distance).
   - Tap **Capture Photo**.
   - Verify the freeze-frame snapshot appears with **Retake** and **Use Photo** action buttons.
6. **Retake Verification**:
   - Tap **Retake**.
   - Verify the live camera stream resumes immediately.
7. **Submission**:
   - Tap **Capture Photo**, then tap **Use Photo & Preprocess**.
   - Verify the base64-encoded frame is posted to `/api/images/upload-base64`.
   - Verify the preprocessed 224×224 image preview appears.

---

## 3. Module 2: Soil Laboratory Form Validation

1. Enter test values:
   - Nitrogen: `280` kg/ha
   - Phosphorus: `45` kg/ha
   - Potassium: `320` kg/ha
   - pH: `6.8`
   - Organic Matter: `2.4` %
   - Moisture: `32` %
2. Tap **Submit & Analyze Soil**.
3. Verify:
   - Overall Soil Health Score is calculated and displayed in the score circle.
   - All nutrient status pills appear with clear green/yellow/red indicators.
   - Actionable recommendations (lime, gypsum, compost, biofertilizers) display correctly.
   - High-compatibility crops list renders clearly.

---

## 4. Mobile Responsiveness & Touch Targets

- [ ] Verify that all action buttons have a minimum touch height of **44px** for easy tapping.
- [ ] Verify the bottom mobile navigation bar allows one-tap switching between Home, Analyze, Farms, Advisory, and Reports.
- [ ] Verify forms do not trigger unintended zooming on Android (font-size >= 14px on all inputs).
- [ ] Verify horizontal scrolling is enabled only inside the data tables (`.table-responsive`) and does not break page width.

---

## 5. Multilingual Language Switching

- [ ] Tap the language selector in the top bar.
- [ ] Select **தமிழ் (Tamil)**:
  - Verify labels switch to Tamil (மண் ஆய்வு, என் பண்ணைகள், பயிர் வழிகாட்டல்).
  - Verify scientific units (`kg/ha`, `pH`, `%`) remain unchanged and accurate.
- [ ] Select **हिंदी (Hindi)** and **ಕನ್ನಡ (Kannada)** and verify label rendering.

---

## 6. Reports & PDF Generation on Android

1. Navigate to the **Reports** tab.
2. Tap **🖨️ Print / Save as PDF**.
3. Verify the Android Print spooler opens.
4. Verify navigation bars and action buttons are cleanly hidden in print preview.
5. Save the PDF to the device downloads folder and verify legibility.
