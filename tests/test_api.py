import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Kejafi" in response.json()["service"]

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_risk_charlotte():
    """Test Charlotte case study from paper."""
    response = client.get(
        "/risk/Charlotte?horizon_years=1.0&confidence=0.95",
        headers={"Authorization": "Bearer kejafi_demo_2024"}
    )
    assert response.status_code == 200
    data = response.json()
    
    assert "ou_parameters" in data
    assert "risk_metrics" in data
    
    ou = data["ou_parameters"]
    assert ou["kappa"] > 0
    assert ou["sigma"] > 0
    
    risk = data["risk_metrics"]
    assert -1.0 < risk["var_95"] < 0
    assert risk["cvar_95"] <= risk["var_95"]

def test_stress_scenarios():
    for scenario in ["BASE_CASE", "COVID_SHOCK", "GFC_2008"]:
        response = client.get(
            f"/stress/Charlotte?scenario={scenario}",
            headers={"Authorization": "Bearer kejafi_demo_2024"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["scenario"] == scenario
        assert "stress_results" in data

def test_cap_rate():
    response = client.get(
        "/cap-rate/Charlotte",
        headers={"Authorization": "Bearer kejafi_demo_2024"}
    )
    assert response.status_code == 200
    data = response.json()
    assert 0.03 < data["final_cap_rate"] < 0.09
    assert data["elasticity_bucket"] in ["VERY_INELASTIC", "INELASTIC", "MODERATE", "ELASTIC"]

def test_compare():
    response = client.get(
        "/compare?metros=Charlotte&metros=Atlanta&metros=Miami",
        headers={"Authorization": "Bearer kejafi_demo_2024"}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 3
    assert len(data["ranking"]) == 3

def test_unauthorized():
    response = client.get("/risk/Charlotte")
    assert response.status_code == 401