import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# --- 1. Knowledge Base (Rule-Based) ---
KNOWLEDGE_BASE_RULES = [
    # Emergency preparedness
    ({"num_emergency_drills_this_year": lambda x: x < 3, "school_type": "elementary"},
     "RECOMMEND_ACTION", "Conduct at least one more emergency drill."),
    ({"num_emergency_drills_this_year": lambda x: x == 0},
     "ALERT", "No emergency drills conducted this year."),
    ({"active_shooter_drill_conducted_last_month": "no"},
     "ALERT", "Critical: Active shooter drill overdue."),
    ({"visitor_log_protocol": "incomplete"}, "RECOMMEND_ACTION",
     "Ensure visitor logs are complete and up-to-date."),
    ({"has_perimeter_fence": "no"}, "ALERT",
     "School perimeter is unfenced. High security risk."),
    ({"has_cctv_cameras": "no"}, "RECOMMEND_ACTION",
     "Install CCTV cameras for monitoring entrances and hallways."),
    ({"school_type": "high", "num_emergency_drills_this_year": lambda x: x < 2},
     "RECOMMEND_ACTION", "High schools should conduct at least 2 drills annually."),

    # Fire safety
    ({"fire_extinguisher_count": lambda x: x < 2}, "ALERT",
     "Insufficient fire extinguishers in the school."),
    ({"fire_alarm_functional": "no"}, "ALERT", "Fire alarm is non-functional."),
    ({"flammable_materials_stored_properly": "no"}, "RECOMMEND_ACTION",
     "Store flammable materials in safe, designated areas."),

    # First aid & medical
    ({"first_aid_kits": lambda x: x == 0},
     "ALERT", "No first aid kits available."),
    ({"nurse_on_staff": "no"}, "RECOMMEND_ACTION",
     "Employ or contract a school nurse."),
    ({"defibrillator_available": "no"}, "RECOMMEND_ACTION",
     "Install a defibrillator in case of cardiac emergencies."),

    # Infrastructure
    ({"emergency_exits_clear": "no"}, "ALERT", "Emergency exits are blocked."),
    ({"playground_equipment_condition": "poor"}, "RECOMMEND_ACTION",
     "Repair or replace unsafe playground equipment."),
    ({"building_structural_integrity": "weak"}, "ALERT",
     "School building has structural weaknesses."),
    ({"classroom_density": lambda x: x > 40}, "RECOMMEND_ACTION",
     "Classrooms are overcrowded. Reduce student density."),

    # Technology & cyber
    ({"internet_security_protocols": "weak"}, "ALERT",
     "Weak cybersecurity protocols in place."),
    ({"student_data_protection_policy": "no"}, "RECOMMEND_ACTION",
     "Implement a data protection policy for student records."),

    # Transportation
    ({"school_bus_maintenance": "poor"}, "ALERT",
     "School buses are unsafe for student transport."),
    ({"bus_driver_background_checks": "incomplete"}, "RECOMMEND_ACTION",
     "Ensure all bus drivers undergo background checks."),

    # Community engagement
    ({"parent_teacher_meetings": lambda x: x < 2}, "RECOMMEND_ACTION",
     "Conduct more parent-teacher engagement sessions."),
    ({"community_safety_drills": "no"}, "RECOMMEND_ACTION",
     "Organize joint safety drills with community responders."),
]


# --- 2. Inference Engine (Forward Chaining) ---
def run_rule_based_inference(facts):
    alerts = []
    recommendations = []
    fired_rules_indices = set()

    for i, (conditions, consequent_type, message) in enumerate(KNOWLEDGE_BASE_RULES):
        all_conditions_met = True
        for fact_key, condition_val in conditions.items():
            if fact_key not in facts:
                all_conditions_met = False
                break

            fact_value = facts[fact_key]

            if callable(condition_val):
                if not condition_val(fact_value):
                    all_conditions_met = False
                    break
            else:
                if fact_value != condition_val:
                    all_conditions_met = False
                    break

        if all_conditions_met and i not in fired_rules_indices:
            fired_rules_indices.add(i)
            if consequent_type == "ALERT":
                alerts.append(message)
            elif consequent_type == "RECOMMEND_ACTION":
                recommendations.append(message)

    return alerts, recommendations


# --- 3. Fuzzy Logic Component ---
def setup_fuzzy_controller():
    supervision = ctrl.Antecedent(np.arange(0, 11, 1), 'supervision')
    maintenance = ctrl.Antecedent(np.arange(0, 11, 1), 'maintenance')
    engagement = ctrl.Antecedent(np.arange(0, 11, 1), 'engagement')
    cyber_prep = ctrl.Antecedent(np.arange(0, 11, 1), 'cyber_prep')
    plan_clarity = ctrl.Antecedent(np.arange(0, 11, 1), 'plan_clarity')

    overall_safety = ctrl.Consequent(np.arange(0, 101, 1), 'overall_safety')

    supervision.automf(3)
    maintenance.automf(3)
    engagement.automf(3)
    cyber_prep.automf(3)
    plan_clarity.automf(3)

    overall_safety['unsafe'] = fuzz.trimf(overall_safety.universe, [0, 0, 50])
    overall_safety['moderate'] = fuzz.trimf(
        overall_safety.universe, [30, 60, 90])
    overall_safety['safe'] = fuzz.trimf(
        overall_safety.universe, [70, 100, 100])

    rule1 = ctrl.Rule(supervision['poor'] |
                      maintenance['poor'], overall_safety['unsafe'])
    rule2 = ctrl.Rule(supervision['average'] & maintenance['average'] & engagement['average'],
                      overall_safety['moderate'])
    rule3 = ctrl.Rule(supervision['good'] & maintenance['good'] & engagement['good']
                      & cyber_prep['good'] & plan_clarity['good'], overall_safety['safe'])
    rule4 = ctrl.Rule(cyber_prep['poor'] |
                      plan_clarity['poor'], overall_safety['unsafe'])

    system = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
    return ctrl.ControlSystemSimulation(system)


# --- 4. GUI Interface (Streamlit) ---
st.set_page_config(layout="wide", page_title="School Safety Audit System")

st.title("🛡️ School Safety Audit Pro")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["New Audit", "Audit History", "Knowledge Base"])

# Initialize session history
if "audit_history" not in st.session_state:
    st.session_state.audit_history = []

if page == "New Audit":
    st.subheader("New School Safety Assessment")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Rule-Based Audit Inputs")
        audit_facts = {}
        audit_facts['school_name'] = st.text_input(
            "School Name", "Example High School")
        audit_facts['school_type'] = st.selectbox(
            "School Type", ['elementary', 'middle', 'high'])
        audit_facts['num_emergency_drills_this_year'] = st.number_input(
            "Emergency Drills This Year", min_value=0, value=2)
        audit_facts['has_perimeter_fence'] = st.radio(
            "Has Perimeter Fence?", ['yes', 'no'])
        audit_facts['has_cctv_cameras'] = st.radio(
            "Has CCTV Cameras?", ['yes', 'no'])
        audit_facts['visitor_log_protocol'] = st.selectbox(
            "Visitor Log Protocol", ['complete', 'incomplete'])
        audit_facts['active_shooter_drill_conducted_last_month'] = st.radio(
            "Active Shooter Drill Last Month?", ['yes', 'no'])

        st.markdown("### Fuzzy Logic Risk Assessment")
        fuzzy_inputs = {}
        fuzzy_inputs['supervision'] = st.slider(
            "Student Supervision Quality", 0, 10, 7)
        fuzzy_inputs['maintenance'] = st.slider(
            "Facility Maintenance", 0, 10, 6)
        fuzzy_inputs['engagement'] = st.slider(
            "Community Engagement", 0, 10, 8)
        fuzzy_inputs['cyber_prep'] = st.slider(
            "Cybersecurity Preparedness", 0, 10, 5)
        fuzzy_inputs['plan_clarity'] = st.slider(
            "Emergency Plan Clarity", 0, 10, 7)

        run_audit = st.button("Run Audit ✅")

    with col2:
        if run_audit:
            st.subheader("📊 Audit Results")

            alerts, recommendations = run_rule_based_inference(audit_facts)

            safety_sim = setup_fuzzy_controller()
            for k, v in fuzzy_inputs.items():
                safety_sim.input[k] = v
            safety_sim.compute()
            overall_safety_score = safety_sim.output['overall_safety']

            # Save result to session history
            st.session_state.audit_history.append({
                "school": audit_facts['school_name'],
                "score": overall_safety_score,
                "alerts": alerts,
                "recs": recommendations
            })

            # Display summary
            st.metric("Overall Safety Score",
                      f"{overall_safety_score:.0f}/100")
            if overall_safety_score > 80:
                st.success("✅ School is generally Safe and well-managed.")
            elif overall_safety_score > 60:
                st.warning(
                    "⚠️ School has Moderate Risk areas. Review recommendations.")
            else:
                st.error("🚨 School is at High Risk. Urgent actions required.")

            # Rule-based findings
            st.subheader("Rule-Based Findings")
            if alerts:
                st.error("🚨 Critical Alerts:")
                for a in alerts:
                    st.write(f"- {a}")
            if recommendations:
                st.info("💡 Recommendations:")
                for r in recommendations:
                    st.write(f"- {r}")
            if not alerts and not recommendations:
                st.success("No specific rule-based alerts or recommendations.")

            # Chart
            st.subheader("📈 Safety Assessment Chart")
            fig, ax = plt.subplots()
            categories = list(fuzzy_inputs.keys())
            values = list(fuzzy_inputs.values())
            ax.bar(categories, values)
            ax.set_ylim(0, 10)
            ax.set_ylabel("Score (0-10)")
            ax.set_title("Fuzzy Input Ratings")
            st.pyplot(fig)

elif page == "Audit History":
    st.subheader("📂 Previous Audit Reports")
    if st.session_state.audit_history:
        for i, report in enumerate(st.session_state.audit_history[::-1], 1):
            with st.expander(f"Audit {i}: {report['school']}"):
                st.write(f"**Safety Score:** {report['score']:.0f}/100")
                if report['alerts']:
                    st.error("🚨 Alerts:")
                    for a in report['alerts']:
                        st.write(f"- {a}")
                if report['recs']:
                    st.info("💡 Recommendations:")
                    for r in report['recs']:
                        st.write(f"- {r}")
    else:
        st.info("No previous audits yet.")

elif page == "Knowledge Base":
    st.subheader("System Knowledge Base")
    for i, (conditions, cons_type, msg) in enumerate(KNOWLEDGE_BASE_RULES[:10]):
        st.write(f"**Rule {i+1}:** IF {conditions} THEN {cons_type}: {msg}")
