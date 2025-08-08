# synthetic_agentic/evaluation_agent.py

from sdmetrics.reports.single_table import QualityReport

def evaluate_quality(real_data, synthetic_data, metadata_dict) -> float:
    """Evaluate synthetic data using SDMetrics QualityReport."""
    try:
        report = QualityReport()
        # report.generate(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata_dict)
        report.generate(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata_dict.to_dict())
        score = report.get_score()
        report.save("quality_report_agentic.html")
        print(f"📊 Quality Score: {score * 100:.2f}% — report saved as 'quality_report_agentic.html'")
        return score
    except Exception as e:
        print(f"[✘] Evaluation failed: {e}")
        return 0.0
