import axios from "axios";

// Change this to your deployed server URL.
// For local dev with Expo Go: use your machine's local IP (not localhost).
// e.g. "http://192.168.1.42:8000"
// For production: "https://your-server.com"
const BASE_URL = process.env.EXPO_PUBLIC_API_URL || "http://192.168.1.100:8000";

const client = axios.create({
  baseURL: BASE_URL,
  timeout: 120_000, // 2 minutes — inference can be slow on CPU
});

/**
 * Run damage assessment on a pre/post image pair.
 *
 * @param {object} params
 * @param {string} params.postImageUri  - local file:// URI for post-disaster image
 * @param {string} [params.preImageUri] - local file:// URI for pre-disaster image (optional)
 * @param {number} [params.lat]         - latitude (decimal degrees)
 * @param {number} [params.lon]         - longitude (decimal degrees)
 * @param {number} [params.latMin]      - bounding box south edge
 * @param {number} [params.lonMin]      - bounding box west edge
 * @param {number} [params.latMax]      - bounding box north edge
 * @param {number} [params.lonMax]      - bounding box east edge
 * @returns {Promise<AnalysisResult>}
 */
export async function analyzeImages({
  postImageUri,
  preImageUri,
  lat,
  lon,
  latMin,
  lonMin,
  latMax,
  lonMax,
}) {
  const form = new FormData();

  form.append("post_image", {
    uri: postImageUri,
    name: "post.jpg",
    type: "image/jpeg",
  });

  if (preImageUri) {
    form.append("pre_image", {
      uri: preImageUri,
      name: "pre.jpg",
      type: "image/jpeg",
    });
  }

  if (lat != null) form.append("lat", String(lat));
  if (lon != null) form.append("lon", String(lon));
  if (latMin != null) form.append("lat_min", String(latMin));
  if (lonMin != null) form.append("lon_min", String(lonMin));
  if (latMax != null) form.append("lat_max", String(latMax));
  if (lonMax != null) form.append("lon_max", String(lonMax));

  const response = await client.post("/analyze", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return response.data;
}

export async function checkHealth() {
  const response = await client.get("/health");
  return response.data;
}
