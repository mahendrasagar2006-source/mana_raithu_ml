/* ===========================================================
   Mana Raithu Nestham — English / Telugu toggle
   Static UI strings are looked up by data-i18n key.
   Dynamic values (state names, crop names, etc. that come
   from the backend/Jinja) are translated by matching their
   English text against KNOWN_TERMS — anything not found is
   left in English rather than breaking.
   =========================================================== */

const UI_TEXT = {
  en: {
    nav_home: "Home",
    headline_index: "Know what to grow, before you sow",
    subtitle_index: "A quick field check for farmers in Telangana & Andhra Pradesh",
    sec1_heading: "Location & season",
    sec2_heading: "Soil & water",
    sec3_heading: "Crop history",
    sec4_heading: "Land & weather",
    lbl_state: "State",
    lbl_season: "Season",
    lbl_soil_type: "Soil type",
    lbl_water: "Water availability",
    lbl_previous_crop: "Previous crop",
    lbl_land_area: "Land area (acres)",
    lbl_temperature: "Temperature (°C)",
    lbl_rainfall: "Rainfall (mm)",
    btn_submit: "Get recommendation",

    headline_result: "Your field, decoded",
    subtitle_result: "Here's what suits the field you described",
    eyebrow: "Recommended crop",
    fert_heading: "Fertilizer dose (kg/acre)",
    lbl_nitrogen: "Nitrogen",
    lbl_phosphorous: "Phosphorous",
    lbl_potassium: "Potassium",
    alt_heading: "Other crops worth considering",
    summary_heading: "Field summary",
    lbl_state_sum: "State",
    lbl_soil_sum: "Soil type",
    lbl_season_sum: "Season",
    btn_again: "Try another field"
  },
  te: {
    nav_home: "హోమ్",
    headline_index: "విత్తే ముందే ఏం పండించాలో తెలుసుకోండి",
    subtitle_index: "తెలంగాణ & ఆంధ్రప్రదేశ్ రైతుల కోసం త్వరిత పొలం పరిశీలన",
    sec1_heading: "ప్రాంతం & సీజన్",
    sec2_heading: "నేల & నీరు",
    sec3_heading: "పంట చరిత్ర",
    sec4_heading: "భూమి & వాతావరణం",
    lbl_state: "రాష్ట్రం",
    lbl_season: "సీజన్",
    lbl_soil_type: "నేల రకం",
    lbl_water: "నీటి లభ్యత",
    lbl_previous_crop: "మునుపటి పంట",
    lbl_land_area: "భూమి విస్తీర్ణం (ఎకరాలు)",
    lbl_temperature: "ఉష్ణోగ్రత (°C)",
    lbl_rainfall: "వర్షపాతం (మి.మీ)",
    btn_submit: "సిఫారసు పొందండి",

    headline_result: "మీ పొలం విశ్లేషణ",
    subtitle_result: "మీరు తెలిపిన పొలానికి ఇది సరిపోతుంది",
    eyebrow: "సిఫారసు చేసిన పంట",
    fert_heading: "ఎరువుల మోతాదు (కిలో/ఎకరం)",
    lbl_nitrogen: "నత్రజని",
    lbl_phosphorous: "భాస్వరం",
    lbl_potassium: "పొటాషియం",
    alt_heading: "పరిగణించదగిన ఇతర పంటలు",
    summary_heading: "పొలం సారాంశం",
    lbl_state_sum: "రాష్ట్రం",
    lbl_soil_sum: "నేల రకం",
    lbl_season_sum: "సీజన్",
    btn_again: "మరో పొలం ప్రయత్నించండి"
  }
};

/* English -> Telugu for values that come from the backend
   (state names, seasons, soil types, crop names). Add more
   entries here any time your model/dataset gains new values —
   anything missing just stays in English, it won't break. */
const KNOWN_TERMS = {
  // states
  "Telangana": "తెలంగాణ",
  "Andhra Pradesh": "ఆంధ్రప్రదేశ్",
  // seasons
  "Kharif": "ఖరీఫ్",
  "Rabi": "రబీ",
  "Zaid": "జైద్",
  "Summer": "వేసవి",
  "Winter": "శీతాకాలం",
  // soil types
  "Black": "నల్ల నేల",
  "Red": "ఎర్ర నేల",
  "Alluvial": "ఒండ్రు నేల",
  "Sandy": "ఇసుక నేల",
  "Clay": "బంక నేల",
  "Loamy": "లోమీ నేల",
  // water availability
  "Rainfed": "వర్షాధారం",
  "Irrigated": "సాగునీటి",
  "Low": "తక్కువ",
  "Medium": "మధ్యస్థం",
  "High": "అధికం",
  // crops
  "Cotton": "పత్తి",
  "Rice": "వరి",
  "Maize": "మొక్కజొన్న",
  "Wheat": "గోధుమ",
  "Sugarcane": "చెరకు",
  "Groundnut": "వేరుశనగ",
  "Chilli": "మిర్చి",
  "Chillies": "మిర్చి",
  "Turmeric": "పసుపు",
  "Soybean": "సోయాబీన్",
  "Jowar": "జొన్న",
  "Bajra": "సజ్జ",
  "Ragi": "రాగి",
  "Sesame": "నువ్వులు",
  "Sunflower": "పొద్దుతిరుగుడు",
  "Gram": "శనగ",
  "Bengal Gram": "శనగ",
  "Redgram": "కంది",
  "Red Gram": "కంది",
  "Blackgram": "మినుము",
  "Black Gram": "మినుము",
  "Greengram": "పెసర",
  "Green Gram": "పెసర",
  "Banana": "అరటి",
  "Mango": "మామిడి",
  "Tobacco": "పొగాకు",
  "Onion": "ఉల్లిపాయ",
  "Tomato": "టమాటా",
  "Groundnuts": "వేరుశనగ",
  "Sorghum": "జొన్న",
  "Millet": "సజ్జ"
};

function translateValue(text, lang){
  if(lang !== 'te') return text;
  return KNOWN_TERMS[text] || text;
}

function applyLanguage(lang){
  document.documentElement.setAttribute('lang', lang === 'te' ? 'te' : 'en');

  // static UI strings
  document.querySelectorAll('[data-i18n]').forEach(function(el){
    const key = el.getAttribute('data-i18n');
    const dict = UI_TEXT[lang] || UI_TEXT.en;
    if(dict[key]) el.textContent = dict[key];
  });

  // <select> options (state, season, soil type, water, previous crop)
  document.querySelectorAll('select option').forEach(function(opt){
    if(!opt.dataset.enText) opt.dataset.enText = opt.textContent.trim();
    opt.textContent = translateValue(opt.dataset.enText, lang);
  });

  // dynamic backend values shown as plain text (crop name, bar names, summary values)
  document.querySelectorAll('.i18n-value').forEach(function(el){
    if(!el.dataset.enText) el.dataset.enText = el.textContent.trim();
    el.textContent = translateValue(el.dataset.enText, lang);
  });

  document.querySelectorAll('.lang-btn').forEach(function(btn){
    const active = btn.dataset.lang === lang;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-pressed', active ? 'true' : 'false');
  });

  try{ localStorage.setItem('mrn_lang', lang); }catch(e){ /* ignore */ }
}

document.addEventListener('DOMContentLoaded', function(){
  document.querySelectorAll('.lang-btn').forEach(function(btn){
    btn.addEventListener('click', function(){ applyLanguage(btn.dataset.lang); });
  });
  let saved = 'en';
  try{ saved = localStorage.getItem('mrn_lang') || 'en'; }catch(e){ /* ignore */ }
  applyLanguage(saved);
});