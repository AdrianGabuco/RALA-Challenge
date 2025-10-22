import React, { useState, useEffect, useCallback } from "react";
import { MapContainer, TileLayer, ImageOverlay } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

function App() {
  const [metadata, setMetadata] = useState(null);
  const [radarUrl, setRadarUrl] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [bounds, setBounds] = useState([
    [20.0, -130.0],
    [55.0, -60.0],
  ]);

  const BACKEND = process.env.REACT_APP_BACKEND || "http://127.0.0.1:8000";

  const fetchMetadata = useCallback(async () => {
  try {
    const res = await fetch(`${BACKEND}/api/radar/metadata/`);
    const data = await res.json();
    setRadarUrl(data.image_url);
    setLastUpdated(data.last_updated);
    setBounds([
      [data.bounds[0], data.bounds[1]],
      [data.bounds[2], data.bounds[3]],
    ]);
  } catch (err) {
    console.error("Error fetching radar metadata:", err);
  }
}, [BACKEND]);

  useEffect(() => {
  fetchMetadata();
  const interval = setInterval(fetchMetadata, 5 * 60 * 1000); 
  return () => clearInterval(interval);
}, [fetchMetadata]);

  return (
    <div className="app-container">
      <div className="header bg-dark text-white p-2 text-center">
        <h3>MRMS Reflectivity at Lowest Altitude (RALA)</h3>
        {lastUpdated && <p>Last Updated: {lastUpdated}</p>}
      </div>

      <MapContainer
        center={[37.5, -95.0]}
        zoom={4}
        scrollWheelZoom={true}
        style={{ height: "90vh", width: "100%" }}
      >
        <TileLayer
          attribution="© OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {radarUrl && <ImageOverlay url={radarUrl} bounds={bounds} opacity={0.7} />}
      </MapContainer>

      <div className="footer text-center p-2 bg-light">
        <small>
          Live radar data provided by NOAA MRMS • Built with React + Django + Leaflet
        </small>
      </div>
    </div>
  );
}

export default App;