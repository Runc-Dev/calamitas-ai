export const DAMAGE_COLORS = {
  0: "#000000", // background
  1: "#00c800", // no_damage   — green
  2: "#ffe600", // minor_damage — yellow
  3: "#ff8000", // major_damage — orange
  4: "#dc0000", // destroyed   — red
  5: "#800080", // unclassified — purple
};

export const DAMAGE_LABELS = {
  0: "Background",
  1: "No Damage",
  2: "Minor Damage",
  3: "Major Damage",
  4: "Destroyed",
  5: "Unclassified",
};

export const DAMAGE_LABELS_TR = {
  0: "Arka plan",
  1: "Sağlam",
  2: "Az hasar",
  3: "Ağır hasar",
  4: "Yıkık",
  5: "Belirsiz",
};

export const PRIORITY_COLOR = (score) => {
  if (score >= 8) return "#dc0000";
  if (score >= 5) return "#ff8000";
  if (score >= 2) return "#ffe600";
  return "#00c800";
};

export const APP_RED = "#c62828";
export const APP_BG = "#f5f5f5";
