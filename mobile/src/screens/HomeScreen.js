import React, { useState } from "react";
import {
  Alert,
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  ActivityIndicator,
  Platform,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import * as Location from "expo-location";
import { analyzeImages } from "../api/client";
import { APP_RED, APP_BG } from "../utils/colors";

export default function HomeScreen({ navigation }) {
  const [postUri, setPostUri]   = useState(null);
  const [preUri,  setPreUri]    = useState(null);
  const [coords,  setCoords]    = useState(null); // {lat, lon}
  const [loading, setLoading]   = useState(false);

  // ── Image picking ───────────────────────────────────────────────────────────

  async function pickImage(setter) {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== "granted") {
      Alert.alert("Permission required", "Photo library access is needed.");
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.85,
    });
    if (!result.canceled) setter(result.assets[0].uri);
  }

  async function takePhoto(setter) {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== "granted") {
      Alert.alert("Permission required", "Camera access is needed.");
      return;
    }
    const result = await ImagePicker.launchCameraAsync({ quality: 0.85 });
    if (!result.canceled) setter(result.assets[0].uri);
  }

  // ── GPS ─────────────────────────────────────────────────────────────────────

  async function getLocation() {
    const { status } = await Location.requestForegroundPermissionsAsync();
    if (status !== "granted") {
      Alert.alert("Permission required", "Location access is needed.");
      return;
    }
    try {
      const loc = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.High });
      setCoords({ lat: loc.coords.latitude, lon: loc.coords.longitude });
      Alert.alert(
        "GPS acquired",
        `${loc.coords.latitude.toFixed(5)}, ${loc.coords.longitude.toFixed(5)}`
      );
    } catch {
      Alert.alert("Error", "Could not get GPS location.");
    }
  }

  // ── Analyze ─────────────────────────────────────────────────────────────────

  async function runAnalysis() {
    if (!postUri) {
      Alert.alert("Required", "Please select a post-disaster image.");
      return;
    }
    setLoading(true);
    try {
      const result = await analyzeImages({
        postImageUri: postUri,
        preImageUri:  preUri,
        lat:  coords?.lat,
        lon:  coords?.lon,
      });
      navigation.navigate("Result", { result, postUri, preUri });
    } catch (err) {
      Alert.alert("Analysis failed", err.message || "Server error. Check API URL.");
    } finally {
      setLoading(false);
    }
  }

  // ── UI ──────────────────────────────────────────────────────────────────────

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>🏚️ AFETSONAR</Text>
      <Text style={styles.subtitle}>Disaster Damage Assessment</Text>

      {/* Post image */}
      <SectionLabel text="Post-disaster image *" />
      <ImageSlot uri={postUri} placeholder="Select post-disaster photo" />
      <View style={styles.row}>
        <SmallBtn label="📁 Gallery" onPress={() => pickImage(setPostUri)} />
        <SmallBtn label="📷 Camera" onPress={() => takePhoto(setPostUri)} />
      </View>

      {/* Pre image */}
      <SectionLabel text="Pre-disaster image (optional)" />
      <ImageSlot uri={preUri} placeholder="Select pre-disaster photo (optional)" />
      <View style={styles.row}>
        <SmallBtn label="📁 Gallery" onPress={() => pickImage(setPreUri)} />
        <SmallBtn label="📷 Camera" onPress={() => takePhoto(setPreUri)} />
        {preUri && <SmallBtn label="✕ Clear" onPress={() => setPreUri(null)} color="#888" />}
      </View>

      {/* GPS */}
      <SectionLabel text="Location (optional)" />
      <TouchableOpacity style={styles.gpsBtn} onPress={getLocation}>
        <Text style={styles.gpsBtnText}>
          {coords
            ? `📍 ${coords.lat.toFixed(4)}, ${coords.lon.toFixed(4)}`
            : "📍 Use device GPS"}
        </Text>
      </TouchableOpacity>
      {coords && (
        <TouchableOpacity onPress={() => setCoords(null)}>
          <Text style={styles.clearLink}>Clear GPS</Text>
        </TouchableOpacity>
      )}

      {/* Analyze */}
      <TouchableOpacity
        style={[styles.analyzeBtn, loading && styles.analyzeBtnDisabled]}
        onPress={runAnalysis}
        disabled={loading}
      >
        {loading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.analyzeBtnText}>🔍 Analyze Damage</Text>
        )}
      </TouchableOpacity>

      {loading && (
        <Text style={styles.loadingNote}>
          Running AI model… this may take 30–120 seconds.
        </Text>
      )}
    </ScrollView>
  );
}

function SectionLabel({ text }) {
  return <Text style={styles.sectionLabel}>{text}</Text>;
}

function ImageSlot({ uri, placeholder }) {
  if (uri) {
    return <Image source={{ uri }} style={styles.imageSlot} />;
  }
  return (
    <View style={[styles.imageSlot, styles.imagePlaceholder]}>
      <Text style={styles.placeholderText}>{placeholder}</Text>
    </View>
  );
}

function SmallBtn({ label, onPress, color }) {
  return (
    <TouchableOpacity
      style={[styles.smallBtn, color ? { backgroundColor: color } : {}]}
      onPress={onPress}
    >
      <Text style={styles.smallBtnText}>{label}</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  container:    { flex: 1, backgroundColor: APP_BG },
  content:      { padding: 16, paddingBottom: 40 },
  title:        { fontSize: 28, fontWeight: "900", color: APP_RED, textAlign: "center", marginTop: 8 },
  subtitle:     { fontSize: 13, color: "#666", textAlign: "center", marginBottom: 20 },
  sectionLabel: { fontSize: 13, fontWeight: "600", color: "#333", marginTop: 16, marginBottom: 6 },
  imageSlot:    { width: "100%", height: 180, borderRadius: 10, backgroundColor: "#e0e0e0" },
  imagePlaceholder: { alignItems: "center", justifyContent: "center" },
  placeholderText:  { color: "#999", fontSize: 13 },
  row:          { flexDirection: "row", gap: 8, marginTop: 8 },
  smallBtn:     { flex: 1, backgroundColor: "#455a64", borderRadius: 8, paddingVertical: 8, alignItems: "center" },
  smallBtnText: { color: "#fff", fontSize: 13, fontWeight: "600" },
  gpsBtn:       { backgroundColor: "#1565c0", borderRadius: 10, paddingVertical: 12, alignItems: "center", marginTop: 4 },
  gpsBtnText:   { color: "#fff", fontSize: 14, fontWeight: "600" },
  clearLink:    { textAlign: "center", color: "#888", fontSize: 12, marginTop: 4 },
  analyzeBtn:   { backgroundColor: APP_RED, borderRadius: 12, paddingVertical: 16, alignItems: "center", marginTop: 28 },
  analyzeBtnDisabled: { opacity: 0.6 },
  analyzeBtnText: { color: "#fff", fontSize: 17, fontWeight: "800" },
  loadingNote:  { textAlign: "center", color: "#666", fontSize: 12, marginTop: 10 },
});
