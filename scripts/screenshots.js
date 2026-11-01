const { chromium } = require("playwright");
const path = require("path");
const fs = require("fs");

const SCREENSHOTS_DIR = path.resolve(__dirname, "../docs/screenshots");

// Allow environment variable overrides for container vs host execution
const STREAMLIT_URL = process.env.STREAMLIT_URL || "http://streamlit:8501";
const API_URL = process.env.API_URL || "http://api:8000";
const MLFLOW_URL = process.env.MLFLOW_URL || "http://mlflow:5000";
const AIRFLOW_URL = process.env.AIRFLOW_URL || "http://airflow:8080";
const AIRFLOW_USER = process.env.AIRFLOW_USER || "admin";
const AIRFLOW_PASSWORD = process.env.AIRFLOW_PASSWORD || "admin";

async function run() {
  if (!fs.existsSync(SCREENSHOTS_DIR)) {
    fs.mkdirSync(SCREENSHOTS_DIR, { recursive: true });
  }

  console.log("Launching Chromium browser...");
  const browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });

  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 2, // HiDPI crisp screenshots
  });

  const page = await context.newPage();

  try {
    // -------------------------------------------------------------
    // 1. STREAMLIT: Initial UI
    // -------------------------------------------------------------
    console.log(`[1/6] Capturing Streamlit Home (${STREAMLIT_URL})...`);
    await page.goto(STREAMLIT_URL, { waitUntil: "networkidle", timeout: 30000 });
    await page.waitForTimeout(2000);
    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, "01-streamlit-home.png"),
      fullPage: true,
    });
    console.log("  -> Saved 01-streamlit-home.png");

    // -------------------------------------------------------------
    // 2. STREAMLIT: High Risk Prediction Flow
    // -------------------------------------------------------------
    console.log("[2/6] Triggering High Risk prediction in Streamlit...");
    // Streamlit inputs: Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age
    const inputs = await page.$$("input[type='number']");
    if (inputs.length >= 7) {
      // High risk values
      const highValues = ["185.0", "85.0", "38.0", "280.0", "36.5", "0.85", "54.0"];
      for (let i = 0; i < highValues.length; i++) {
        await inputs[i].click({ clickCount: 3 });
        await inputs[i].fill(highValues[i]);
        await inputs[i].press("Enter");
      }
    }
    // Click "Predict risk" button
    const predictBtn = page.getByRole("button", { name: /predict risk/i });
    if (await predictBtn.count() > 0) {
      await predictBtn.first().click();
      await page.waitForTimeout(3000);
    }
    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, "02-streamlit-high-risk.png"),
      fullPage: true,
    });
    console.log("  -> Saved 02-streamlit-high-risk.png");

    // -------------------------------------------------------------
    // 3. STREAMLIT: Low Risk Prediction Flow
    // -------------------------------------------------------------
    console.log("[3/6] Triggering Low Risk prediction in Streamlit...");
    const lowInputs = await page.$$("input[type='number']");
    if (lowInputs.length >= 7) {
      // Low risk values
      const lowValues = ["88.0", "65.0", "20.0", "65.0", "22.5", "0.18", "26.0"];
      for (let i = 0; i < lowValues.length; i++) {
        await lowInputs[i].click({ clickCount: 3 });
        await lowInputs[i].fill(lowValues[i]);
        await lowInputs[i].press("Enter");
      }
    }
    if (await predictBtn.count() > 0) {
      await predictBtn.first().click();
      await page.waitForTimeout(3000);
    }
    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, "03-streamlit-low-risk.png"),
      fullPage: true,
    });
    console.log("  -> Saved 03-streamlit-low-risk.png");

    // -------------------------------------------------------------
    // 4. FASTAPI: Swagger Interactive API Documentation
    // -------------------------------------------------------------
    console.log(`[4/6] Capturing FastAPI Swagger UI (${API_URL}/docs)...`);
    await page.goto(`${API_URL}/docs`, { waitUntil: "networkidle", timeout: 30000 });
    await page.waitForTimeout(2000);
    // Expand /predict endpoint
    const postPredictOp = page.locator("#operations-default-predict_predict_post .opblock-summary");
    if (await postPredictOp.count() > 0) {
      await postPredictOp.first().click();
      await page.waitForTimeout(1000);
    }
    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, "04-fastapi-swagger.png"),
      fullPage: true,
    });
    console.log("  -> Saved 04-fastapi-swagger.png");

    // -------------------------------------------------------------
    // 5. MLFLOW: Tracking Server UI & Model Registry
    // -------------------------------------------------------------
    console.log(`[5/6] Capturing MLflow UI (${MLFLOW_URL})...`);
    await page.goto(MLFLOW_URL, { waitUntil: "networkidle", timeout: 30000 });
    await page.waitForTimeout(3000);
    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, "05-mlflow-dashboard.png"),
      fullPage: false,
    });
    console.log("  -> Saved 05-mlflow-dashboard.png");

    // Go to Models tab
    try {
      await page.goto(`${MLFLOW_URL}/#/models`, { waitUntil: "networkidle", timeout: 30000 });
      await page.waitForTimeout(3000);
      await page.screenshot({
        path: path.join(SCREENSHOTS_DIR, "06-mlflow-models.png"),
        fullPage: false,
      });
      console.log("  -> Saved 06-mlflow-models.png");
    } catch (e) {
      console.warn("  Could not capture MLflow models page:", e.message);
    }

    // -------------------------------------------------------------
    // 6. AIRFLOW: DAG Orchestration UI
    // -------------------------------------------------------------
    console.log(`[6/6] Capturing Airflow UI (${AIRFLOW_URL})...`);
    try {
      await page.goto(AIRFLOW_URL, { waitUntil: "networkidle", timeout: 30000 });
      await page.waitForTimeout(2000);

      // Check if redirected to login page
      const usernameField = page.locator("input[name='username']");
      if (await usernameField.count() > 0) {
        console.log("  Logging into Airflow web UI...");
        await usernameField.first().fill(AIRFLOW_USER);
        const passField = page.locator("input[name='password']");
        await passField.first().fill(AIRFLOW_PASSWORD);
        await page.locator("button[type='submit'], input[type='submit']").first().click();
        await page.waitForTimeout(3000);
      }

      await page.screenshot({
        path: path.join(SCREENSHOTS_DIR, "07-airflow-dags.png"),
        fullPage: false,
      });
      console.log("  -> Saved 07-airflow-dags.png");

      // Navigate directly to the DAG grid view
      await page.goto(`${AIRFLOW_URL}/dags/diabetes_retraining/grid`, {
        waitUntil: "networkidle",
        timeout: 30000,
      });
      await page.waitForTimeout(3000);
      await page.screenshot({
        path: path.join(SCREENSHOTS_DIR, "08-airflow-dag-runs.png"),
        fullPage: false,
      });
      console.log("  -> Saved 08-airflow-dag-runs.png");
    } catch (e) {
      console.warn("  Could not capture Airflow UI:", e.message);
    }

    console.log("\nAll screenshots captured successfully!");
  } catch (err) {
    console.error("Screenshot capture failed:", err);
  } finally {
    await browser.close();
  }
}

run();
