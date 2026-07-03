import React, { useState } from "react";
import {
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import MapView, { Marker, Circle } from "react-native-maps";
import {
  DAMAGE_COLORS,
  DAMAGE_LABELS,
  PRIORITY_COLOR,
  APP_RED,
  APP_BG,
} from "../utils/colors";

export default function ResultScreen({ route }) {
  const { result, postUri } = route.params;
  const [tab, setTab] = useState("stats"); // "stats" | "buildings" | "map"

  const hasBbox = result.bbox != null;
  const hasBuildings = result.buildings && result.buildings.length > 0;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>📊 Analysis Results</Text>

      {/* Post image + mask overlay */}
      {result.mask_png_b64 ? (
        <Image
          source={{ uri: `data:image/png;base64,${result.mask_png_b64}` }}
          style={styles.maskImage}
          resizeMode="contain"
        />
      ) : null}

      {/* Tab bar */}
      <View style={styles.tabBar}>
        {["stats", "buildings", "map"].map((t) => (
          <TouchableOpacity
            key={t}
            style={[styles.tab, tab === t && styles.tabActive]}
            onPress={() => setTab(t)}
          >
            <Text style={[styles.tabText, tab === t && styles.tabTextActive]}>
              {t === "stats" ? "📊 Stats" : t === "buildings" ? "🏗️ Buildings" : "🗺️ Map"}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Stats tab */}
      {tab === "stats" && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Damage Distribution</Text>
          {result.stats
            .filter((s) => s.class_id > 0)
            .map((s) => (
              <View key={s.class_id} style={styles.statRow}>
                <View
                  style={[
                    styles.colorDot,
                    { backgroundColor: DAMAGE_COLORS[s.class_id] },
                  ]}
                />
                <Text style={styles.statLabel}>{DAMAGE_LABELS[s.class_id]}</Text>
                <Text style={styles.statPct}>{s.percentage.toFixed(1)}%</Text>
                <View style={styles.statBar}>
                  <View
                    style={[
                      styles.statBarFill,
                      {
                        width: `${Math.min(s.percentage * 5, 100)}%`,
                        backgroundColor: DAMAGE_COLORS[s.class_id],
                      },
                    ]}
                  />
                </View>
              </View>
            ))}
        </View>
      )}

      {/* Buildings tab */}
      {tab === "buildings" && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>
            {result.buildings.length} Damaged Buildings
          </Text>
          {!hasBuildings && (
            <Text style={styles.noData}>
              No buildings detected. Add GPS or bounding box for geo-referencing.
            </Text>
          )}
          {result.buildings
            .sort((a, b) => b.priority_score - a.priority_score)
            .map((b) => (
              <View key={b.building_id} style={styles.buildingRow}>
                <View
                  style={[
                    styles.priorityBadge,
                    { backgroundColor: PRIORITY_COLOR(b.priority_score) },
                  ]}
                >
                  <Text style={styles.priorityText}>
                    {b.priority_score.toFixed(1)}
                  </Text>
                </View>
                <View style={styles.buildingInfo}>
                  <Text style={styles.buildingName}>
                    {DAMAGE_LABELS[b.damage_class]} · {b.area_m2.toFixed(0)} m²
                  </Text>
                  {b.lat != null && (
                    <Text style={styles.buildingCoords}>
                      {b.lat.toFixed(4)}, {b.lon.toFixed(4)}
                    </Text>
                  )}
                </View>
              </View>
            ))}
        </View>
      )}

      {/* Map tab */}
      {tab === "map" && (
        <View style={styles.mapContainer}>
          {result.center_lat !== 0 || result.center_lon !== 0 ? (
            <MapView
              style={styles.map}
              initialRegion={{
                latitude:      result.center_lat,
                longitude:     result.center_lon,
                latitudeDelta:  hasBbox
                  ? Math.abs(result.bbox[2] - result.bbox[0]) * 1.5
                  : 0.02,
                longitudeDelta: hasBbox
                  ? Math.abs(result.bbox[3] - result.bbox[1]) * 1.5
                  : 0.02,
              }}
              mapType="satellite"
            >
              {result.buildings
                .filter((b) => b.lat != null && b.lon != null)
                .map((b) => (
                  <Circle
                    key={b.building_id}
                    center={{ latitude: b.lat, longitude: b.lon }}
                    radius={Math.sqrt(b.area_m2 / Math.PI)}
                    strokeColor={DAMAGE_COLORS[b.damage_class]}
                    fillColor={DAMAGE_COLORS[b.damage_class] + "88"}
                    strokeWidth={2}
                  />
                ))}
            </MapView>
          ) : (
            <View style={styles.mapPlaceholder}>
              <Text style={styles.noData}>
                Map requires GPS coordinates or a bounding box.
              </Text>
            </View>
          )}
        </View>
      )}

      {/* Legend */}
      <View style={styles.legend}>
        {[1, 2, 3, 4].map((cls) => (
          <View key={cls} style={styles.legendItem}>
            <View
              style={[styles.legendDot, { backgroundColor: DAMAGE_COLORS[cls] }]}
            />
            <Text style={styles.legendText}>{DAMAGE_LABELS[cls]}</Text>
          </View>
        ))}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container:    { flex: 1, backgroundColor: APP_BG },
  content:      { padding: 16, paddingBottom: 40 },
  title:        { fontSize: 22, fontWeight: "800", color: APP_RED, marginBottom: 12 },
  maskImage:    { width: "100%", height: 220, borderRadius: 10, backgroundColor: "#000", marginBottom: 12 },
  tabBar:       { flexDirection: "row", backgroundColor: "#e0e0e0", borderRadius: 10, marginBottom: 12 },
  tab:          { flex: 1, paddingVertical: 10, alignItems: "center", borderRadius: 10 },
  tabActive:    { backgroundColor: APP_RED },
  tabText:      { fontSize: 13, color: "#555" },
  tabTextActive: { color: "#fff", fontWeight: "700" },
  card:         { backgroundColor: "#fff", borderRadius: 12, padding: 14, marginBottom: 12, elevation: 2 },
  cardTitle:    { fontSize: 15, fontWeight: "700", marginBottom: 10, color: "#333" },
  statRow:      { flexDirection: "row", alignItems: "center", marginBottom: 8 },
  colorDot:     { width: 12, height: 12, borderRadius: 6, marginRight: 8 },
  statLabel:    { flex: 1, fontSize: 13, color: "#444" },
  statPct:      { fontSize: 13, fontWeight: "600", width: 48, textAlign: "right" },
  statBar:      { width: 60, height: 6, backgroundColor: "#eee", borderRadius: 3, marginLeft: 8 },
  statBarFill:  { height: 6, borderRadius: 3 },
  buildingRow:  { flexDirection: "row", alignItems: "center", paddingVertical: 8, borderBottomWidth: 1, borderBottomColor: "#eee" },
  priorityBadge: { width: 42, height: 42, borderRadius: 21, alignItems: "center", justifyContent: "center", marginRight: 12 },
  priorityText: { color: "#fff", fontWeight: "800", fontSize: 13 },
  buildingInfo: { flex: 1 },
  buildingName: { fontSize: 13, fontWeight: "600", color: "#333" },
  buildingCoords: { fontSize: 11, color: "#888", marginTop: 2 },
  noData:       { color: "#999", textAlign: "center", padding: 16, fontStyle: "italic" },
  mapContainer: { height: 380, borderRadius: 12, overflow: "hidden", marginBottom: 12 },
  map:          { flex: 1 },
  mapPlaceholder: { flex: 1, alignItems: "center", justifyContent: "center", backgroundColor: "#e0e0e0" },
  legend:       { flexDirection: "row", flexWrap: "wrap", justifyContent: "center", gap: 12, marginTop: 8 },
  legendItem:   { flexDirection: "row", alignItems: "center", gap: 6 },
  legendDot:    { width: 12, height: 12, borderRadius: 6 },
  legendText:   { fontSize: 11, color: "#555" },
});
