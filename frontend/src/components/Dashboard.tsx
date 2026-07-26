"use client";

import React, { useEffect, useState } from "react";
import { Card, Text, Metric, ProgressCircle, Title, Subtitle, Divider, AreaChart, Badge } from "@tremor/react";
import {
  Search,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  ShieldCheck,
  ShieldAlert,
  Activity,
  Zap,
  Clock,
  Target,
  TrendingUp,
  Info,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface DashboardProps {
  activeTab?: string;
  severityFilter?: string;
}

interface EvalReport {
  classification_report: Record<string, { precision: number; recall: number; f1_score: number; support: number }>;
  confusion_matrix: { labels: string[]; matrix: number[][] };
  binary_detection: {
    true_positives: number;
    true_negatives: number;
    false_positives: number;
    false_negatives: number;
    false_positive_rate: number;
    detection_rate: number;
  };
  overall_accuracy: number;
  macro_avg: { precision: number; recall: number; f1_score: number };
  weighted_avg: { precision: number; recall: number; f1_score: number };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function Dashboard({ activeTab = "Dashboard", severityFilter = "all" }: DashboardProps) {
  const [metrics, setMetrics] = useState<any>(null);
  const [alerts, setAlerts] = useState<any[]>([]);
  const [entityId, setEntityId] = useState<string>("USR_101");
  const [entityData, setEntityData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [debouncedEntityId, setDebouncedEntityId] = useState<string>("USR_101");
  const [entityNotFound, setEntityNotFound] = useState<boolean>(false);
  const [expandedRow, setExpandedRow] = useState<number | null>(null);
  const [evalData, setEvalData] = useState<EvalReport | null>(null);

  // Debounce the search input
  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedEntityId(entityId);
    }, 500);
    return () => clearTimeout(handler);
  }, [entityId]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const metricsRes = await fetch("http://127.0.0.1:8000/api/metrics");
        if (!metricsRes.ok) throw new Error("Failed to fetch metrics data.");
        const metricsData = await metricsRes.json();
        setMetrics(metricsData);

        const alertsRes = await fetch("http://127.0.0.1:8000/api/alerts?limit=100");
        if (!alertsRes.ok) throw new Error("Failed to fetch alerts data.");
        const alertsData = await alertsRes.json();
        setAlerts(alertsData);
        
        if (alertsData.length > 0 && !entityId) {
          setEntityId(alertsData[0].entity_id);
        }

        // Fetch evaluation metrics
        try {
          const evalRes = await fetch("http://127.0.0.1:8000/api/evaluation");
          if (evalRes.ok) {
            const evalJson = await evalRes.json();
            setEvalData(evalJson);
          }
        } catch {
          // Evaluation endpoint may not be available — non-blocking
        }
      } catch (err: any) {
        console.error("Error fetching data:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  useEffect(() => {
    if (!debouncedEntityId) return;
    const fetchEntity = async () => {
      try {
        setEntityNotFound(false);
        const res = await fetch(`http://127.0.0.1:8000/api/entity/${debouncedEntityId}`);
        if (res.ok) {
          const data = await res.json();
          setEntityData(data);
        } else if (res.status === 404) {
          setEntityData(null);
          setEntityNotFound(true);
        } else {
          setEntityData(null);
        }
      } catch (err: any) {
        console.error("Error fetching entity data:", err);
      }
    };
    fetchEntity();
  }, [debouncedEntityId]);

  // ── Severity filter logic ──
  const filteredAlerts = alerts.filter((a) => {
    if (severityFilter === "all") return true;
    const score = a.anomaly_score ?? 0;
    if (severityFilter === "critical") return score > 0.8;
    if (severityFilter === "high") return score > 0.6 && score <= 0.8;
    if (severityFilter === "medium") return score > 0.4 && score <= 0.6;
    return true;
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex items-center gap-3 text-blue-600">
          <Activity className="w-5 h-5 animate-pulse" />
          <span className="text-sm font-medium">Loading Assessment Data...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto h-full flex flex-col items-center justify-center pt-20">
        <Card className="max-w-md text-center border border-rose-200 bg-rose-50 text-rose-900">
          <div className="flex items-center justify-center mb-3">
            <AlertTriangle className="w-8 h-8 text-rose-600" />
          </div>
          <Title className="text-rose-900 mb-2 text-xl font-semibold">Connection Error</Title>
          <Subtitle className="text-rose-700">
            {error}. Ensure the FastAPI backend is running on port 8000.
          </Subtitle>
        </Card>
      </div>
    );
  }

  // Map SHAP data for charts
  const shapChartData = (entityData?.shap_attribution || []).map((item: any) => ({
    name: item.feature,
    value: item.impact
  }));

  // Derived metrics
  const threatRate = metrics?.total_events > 0 
    ? ((metrics.critical_alerts / metrics.total_events) * 100).toFixed(1) 
    : "0.0";

  // Synthesize Global Threat Landscape Data
  const globalThreatData = alerts.length > 0 
    ? alerts.slice(0, 20).reverse().map((a, i) => ({
        time: a.timestamp_str || `T-${20 - i}`,
        RiskScore: a.anomaly_score
      }))
    : [];

  // Get severity badge color and label
  const getSeverity = (score: number) => {
    if (score > 0.8) return { color: "rose" as const, label: "CRITICAL", icon: ShieldAlert };
    if (score > 0.6) return { color: "amber" as const, label: "HIGH", icon: AlertTriangle };
    if (score > 0.4) return { color: "yellow" as const, label: "MEDIUM", icon: Activity };
    return { color: "emerald" as const, label: "LOW", icon: ShieldCheck };
  };

  // ════════════════════════════════════════════════════════════════════
  // Tab: Dashboard
  // ════════════════════════════════════════════════════════════════════
  const MetricsRibbon = () => (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-5 mb-8">
      <Card className="flex items-center gap-5 border border-emerald-200 bg-emerald-50 shadow-sm">
        <ProgressCircle value={metrics?.detection_accuracy || 0} size="md" color="emerald">
          <span className="text-[10px] font-bold text-emerald-800">{metrics?.detection_accuracy || 0}%</span>
        </ProgressCircle>
        <div>
          <Text className="text-emerald-700 text-xs font-semibold">Detection Accuracy</Text>
          <Metric className="text-emerald-900 text-2xl font-bold">{metrics?.detection_accuracy || 0}%</Metric>
        </div>
      </Card>
      
      <Card className="flex items-center gap-5 border border-amber-200 bg-amber-50 shadow-sm">
        <ProgressCircle value={Math.min((metrics?.false_positive_rate || 0) * 1000, 100)} size="md" color="amber">
          <span className="text-[10px] font-bold text-amber-800">{((metrics?.false_positive_rate || 0) * 100).toFixed(2)}%</span>
        </ProgressCircle>
        <div>
          <Text className="text-amber-700 text-xs font-semibold">False Positive Rate</Text>
          <Metric className="text-amber-900 text-2xl font-bold">{((metrics?.false_positive_rate || 0) * 100).toFixed(2)}%</Metric>
        </div>
      </Card>

      <Card className="flex items-center gap-5 border border-blue-200 bg-blue-50 shadow-sm">
        <ProgressCircle value={100} size="md" color="blue">
          <Zap className="w-4 h-4 text-blue-600" />
        </ProgressCircle>
        <div>
          <Text className="text-blue-700 text-xs font-semibold">Avg Detection Latency</Text>
          <Metric className="text-blue-900 text-2xl font-bold">{metrics?.latency_ms || 0} ms</Metric>
        </div>
      </Card>

      <Card className="flex items-center gap-5 border border-rose-200 bg-rose-50 shadow-sm">
        <ProgressCircle value={Math.min(Number(threatRate) * 10, 100)} size="md" color="rose">
          <ShieldAlert className="w-4 h-4 text-rose-600" />
        </ProgressCircle>
        <div>
          <Text className="text-rose-700 text-xs font-semibold">Critical Risk Alerts</Text>
          <Metric className="text-rose-900 text-2xl font-bold">{metrics?.critical_alerts || 0}</Metric>
        </div>
      </Card>
    </div>
  );

  const DashboardTab = () => (
    <div>
      <MetricsRibbon />
      <Card className="border border-slate-200 bg-white shadow-sm">
        <Title className="text-slate-900">Global Threat Landscape</Title>
        <Subtitle className="text-slate-500 mb-6">Aggregated risk scores from top anomalous events across the network.</Subtitle>
        {globalThreatData.length > 0 ? (
          <AreaChart
            className="h-72 mt-4"
            data={globalThreatData}
            index="time"
            categories={["RiskScore"]}
            colors={["blue"]}
            valueFormatter={(number) => number.toFixed(3)}
            showGridLines={true}
            showLegend={false}
          />
        ) : (
          <div className="flex items-center justify-center h-72 bg-slate-50 border border-dashed border-slate-200 rounded-lg text-slate-400 text-sm">
            No global threat data available
          </div>
        )}
      </Card>
    </div>
  );

  // ════════════════════════════════════════════════════════════════════
  // Tab: Threat Assessments — with expandable rows
  // ════════════════════════════════════════════════════════════════════
  const ThreatAssessmentsTab = () => (
    <Card className="flex flex-col p-0 overflow-hidden border border-slate-200 bg-white shadow-sm min-h-[600px]">
      <div className="p-6 border-b border-slate-200">
        <div className="flex items-center justify-between">
          <div>
            <Title className="text-slate-900">Ranked Alert Queue</Title>
            <Subtitle className="text-slate-500">Entities sorted by anomalous behavior probability. Click a row to expand.</Subtitle>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <Target className="w-4 h-4 text-blue-600" />
            <span>{filteredAlerts.length} alerts</span>
          </div>
        </div>
      </div>
      
      <div className="overflow-y-auto flex-1">
        <table className="w-full text-sm text-left">
          <thead className="text-[11px] uppercase bg-slate-50 text-slate-500 sticky top-0 border-b border-slate-200 z-10">
            <tr>
              <th className="px-6 py-3 font-semibold w-8"></th>
              <th className="px-4 py-3 font-semibold">Timestamp</th>
              <th className="px-4 py-3 font-semibold">Entity ID</th>
              <th className="px-4 py-3 font-semibold">Severity</th>
              <th className="px-4 py-3 font-semibold">Classification</th>
              <th className="px-4 py-3 font-semibold text-right">Risk Score</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200">
            {!filteredAlerts || filteredAlerts.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-6 py-12 text-center text-slate-500">
                  <div className="flex flex-col items-center gap-2">
                    <ShieldCheck className="w-8 h-8 text-slate-400" />
                    <span>No alerts match current filter criteria</span>
                  </div>
                </td>
              </tr>
            ) : (
              filteredAlerts.map((alert, idx) => {
                const severity = getSeverity(alert.anomaly_score);
                const SevIcon = severity.icon;
                const isExpanded = expandedRow === idx;

                return (
                  <React.Fragment key={idx}>
                    <tr 
                      className="hover:bg-slate-50 transition-colors cursor-pointer"
                      onClick={() => setExpandedRow(isExpanded ? null : idx)}
                    >
                      <td className="px-6 py-3 text-slate-400">
                        {isExpanded
                          ? <ChevronUp className="w-4 h-4 text-blue-600" />
                          : <ChevronDown className="w-4 h-4" />
                        }
                      </td>
                      <td className="px-4 py-3 text-slate-600 whitespace-nowrap font-mono text-xs">{alert.timestamp_str}</td>
                      <td className="px-4 py-3 font-mono font-semibold text-slate-900">{alert.entity_id}</td>
                      <td className="px-4 py-3">
                        <span className="flex items-center gap-1.5">
                          <SevIcon className={`w-3.5 h-3.5 ${
                            severity.label === "CRITICAL" ? "text-rose-600" :
                            severity.label === "HIGH" ? "text-amber-600" :
                            severity.label === "MEDIUM" ? "text-yellow-600" :
                            "text-emerald-600"
                          }`} />
                          <span className={`text-xs font-bold tracking-wider ${
                            severity.label === "CRITICAL" ? "text-rose-600" :
                            severity.label === "HIGH" ? "text-amber-600" :
                            severity.label === "MEDIUM" ? "text-yellow-600" :
                            "text-emerald-600"
                          }`}>
                            {severity.label}
                          </span>
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <Badge color={alert.anomaly_score > 0.8 ? "rose" : "amber"}>
                          {alert.predicted_attack}
                        </Badge>
                      </td>
                      <td className="px-4 py-3 font-semibold text-right text-slate-900 font-mono">
                        {alert.anomaly_score.toFixed(4)}
                      </td>
                    </tr>
                    {/* ── Expanded row: SHAP breakdown + explanation ── */}
                    {isExpanded && (
                      <tr>
                        <td colSpan={6} className="px-0 py-0">
                          <div className="bg-slate-50 border-t border-b border-slate-200 px-10 py-5 space-y-4">
                            {/* Explanation string */}
                            {alert.explanation_string && (
                              <div className="flex items-start gap-3 bg-blue-50/50 border border-blue-200 rounded-lg px-4 py-3">
                                <Info className="w-4 h-4 text-blue-600 mt-0.5 shrink-0" />
                                <p className="text-sm text-blue-800 italic leading-relaxed">
                                  {alert.explanation_string}
                                </p>
                              </div>
                            )}
                            {/* Alert metadata */}
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                              <div>
                                <span className="text-slate-500 block mb-1">Source IP</span>
                                <span className="text-slate-800 font-mono">{alert.source_ip}</span>
                              </div>
                              <div>
                                <span className="text-slate-500 block mb-1">Geo Location</span>
                                <span className="text-slate-800">{alert.geo_location}</span>
                              </div>
                              <div>
                                <span className="text-slate-500 block mb-1">Auth Method</span>
                                <span className="text-slate-800">{alert.auth_method}</span>
                              </div>
                              <div>
                                <span className="text-slate-500 block mb-1">Session Duration</span>
                                <span className="text-slate-800">{alert.session_duration} min</span>
                              </div>
                              <div>
                                <span className="text-slate-500 block mb-1">Resource Accessed</span>
                                <span className="text-slate-800 font-mono text-[11px]">{alert.resource_accessed}</span>
                              </div>
                              <div>
                                <span className="text-slate-500 block mb-1">Device Fingerprint</span>
                                <span className="text-slate-800 font-mono text-[11px]">{alert.device_fingerprint}</span>
                              </div>
                              <div>
                                <span className="text-slate-500 block mb-1">Entity Type</span>
                                <span className="text-slate-800">{alert.entity_type}</span>
                              </div>
                              <div>
                                <span className="text-slate-500 block mb-1">Ground Truth</span>
                                <span className="text-slate-800 font-mono">{alert.label}</span>
                              </div>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </Card>
  );

  // ════════════════════════════════════════════════════════════════════
  // Tab: Incident Reports — entity deep-dive
  // ════════════════════════════════════════════════════════════════════
  const IncidentReportsTab = () => (
    <Card className="flex flex-col border border-slate-200 bg-white shadow-sm min-h-[600px]">
      <Title className="text-slate-900 uppercase tracking-wide text-sm font-bold">Entity Deep-Dive Inspector</Title>
      <Divider className="my-4" />
      
      <div className="mb-6 relative max-w-md">
        <label className="text-[10px] font-semibold text-slate-500 mb-2 block uppercase tracking-widest">Search Entity</label>
        <div className="relative">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input 
            type="text" 
            value={entityId || ""}
            onChange={(e) => setEntityId(e.target.value)}
            className="w-full pl-9 pr-4 py-2.5 bg-white border border-slate-300 rounded-lg text-sm text-slate-950 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-all placeholder:text-slate-400"
            placeholder="e.g. USR_101"
          />
        </div>
      </div>

      {/* Explanation string callout */}
      {entityData?.explanation_string && !entityNotFound && (
        <div className="mb-6 flex items-start gap-3 bg-blue-50 border border-blue-200 rounded-lg px-4 py-3 max-w-3xl">
          <Info className="w-4 h-4 text-blue-600 mt-0.5 shrink-0" />
          <p className="text-sm text-blue-800 italic leading-relaxed">
            {entityData.explanation_string}
          </p>
        </div>
      )}

      <div className="flex-1 overflow-y-auto pr-2 space-y-8">
        
        {/* Historical Scores Area Chart */}
        <div>
          <Subtitle className="text-slate-800 font-medium mb-2 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-blue-600" />
            Entity History View
          </Subtitle>
          {entityNotFound ? (
            <div className="flex items-center justify-center h-60 bg-slate-50 border border-dashed border-slate-200 rounded-lg text-slate-400 text-sm">
              Awaiting complete and valid Entity ID...
            </div>
          ) : !entityData?.historical_scores || entityData.historical_scores.length === 0 ? (
            <div className="flex items-center justify-center h-60 bg-slate-50 border border-dashed border-slate-200 rounded-lg text-slate-400 text-sm">
              No history available
            </div>
          ) : (
            <AreaChart
              className="h-60 mt-2"
              data={entityData.historical_scores}
              index="time"
              categories={["RiskScore"]}
              colors={["blue"]}
              valueFormatter={(number) => number.toFixed(3)}
              showGridLines={true}
              showLegend={false}
            />
          )}
        </div>

        {/* SHAP Attribution */}
        <div>
          <Subtitle className="text-slate-800 font-medium mb-4 flex items-center gap-2">
            <Activity className="w-4 h-4 text-blue-600" />
            Contributing Factors (SHAP Feature Attribution)
          </Subtitle>
          
          {entityNotFound ? (
            <div className="flex items-center justify-center h-40 bg-slate-50 border border-dashed border-slate-200 rounded-lg text-slate-400 text-sm">
              Entity not found in current historical index.
            </div>
          ) : !shapChartData || shapChartData.length === 0 ? (
            <div className="flex items-center justify-center h-40 bg-slate-50 border border-dashed border-slate-200 rounded-lg text-slate-400 text-sm">
              No attribution data available
            </div>
          ) : (
            <div className="space-y-3 mt-4 max-w-3xl">
              {shapChartData.map((item: any, idx: number) => {
                const maxVal = Math.max(...shapChartData.map((d: any) => d.value));
                const percentage = maxVal > 0 ? (item.value / maxVal) * 100 : 0;
                
                return (
                  <div key={idx} className="flex items-center text-sm">
                    <div className="w-52 truncate text-slate-600 mr-4 font-medium font-mono text-xs" title={item.name}>
                      {item.name}
                    </div>
                    <div className="flex-1 flex items-center gap-3">
                      <div className="flex-1 h-7 bg-slate-100 rounded overflow-hidden flex items-center">
                        <div 
                          className="h-full bg-gradient-to-r from-blue-600 to-blue-400 rounded transition-all duration-700" 
                          style={{ width: `${Math.max(percentage, 2)}%` }}
                        />
                      </div>
                      <div className="w-20 text-right font-semibold text-slate-800 font-mono text-xs">
                        {item.value.toFixed(4)}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
      
      <Divider className="my-4" />
      <div className="text-xs text-slate-500 text-center">
        Scores represent SHAP feature attribution magnitudes from the XGBoost classifier via TreeExplainer.
      </div>
    </Card>
  );

  // ════════════════════════════════════════════════════════════════════
  // Tab: Evaluation Metrics — classification report, confusion matrix, FPR
  // ════════════════════════════════════════════════════════════════════
  const EvaluationMetricsTab = () => {
    if (!evalData) {
      return (
        <Card className="border border-slate-200 bg-white shadow-sm flex items-center justify-center min-h-[400px]">
          <div className="text-center text-slate-500">
            <Activity className="w-8 h-8 mx-auto mb-3 text-slate-400" />
            <p>Evaluation data unavailable. Ensure the backend is running.</p>
          </div>
        </Card>
      );
    }

    const classLabels = Object.keys(evalData.classification_report);
    const cm = evalData.confusion_matrix;
    const bd = evalData.binary_detection;

    return (
      <div className="space-y-6">
        {/* Summary Ribbon */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-5">
          <Card decoration="top" decorationColor="emerald" className="border border-slate-200 bg-white shadow-sm">
            <Text className="text-slate-500 text-xs">Overall Accuracy</Text>
            <Metric className="text-slate-900 text-3xl mt-1 font-bold">{(evalData.overall_accuracy * 100).toFixed(2)}%</Metric>
          </Card>
          <Card decoration="top" decorationColor="blue" className="border border-slate-200 bg-white shadow-sm">
            <Text className="text-slate-500 text-xs">Macro F1-Score</Text>
            <Metric className="text-slate-900 text-3xl mt-1 font-bold">{(evalData.macro_avg.f1_score * 100).toFixed(2)}%</Metric>
          </Card>
          <Card decoration="top" decorationColor="amber" className="border border-slate-200 bg-white shadow-sm">
            <Text className="text-slate-500 text-xs">False Positive Rate</Text>
            <Metric className="text-slate-900 text-3xl mt-1 font-bold">{(bd.false_positive_rate * 100).toFixed(3)}%</Metric>
          </Card>
          <Card decoration="top" decorationColor="rose" className="border border-slate-200 bg-white shadow-sm">
            <Text className="text-slate-500 text-xs">Detection Rate (TPR)</Text>
            <Metric className="text-slate-900 text-3xl mt-1 font-bold">{(bd.detection_rate * 100).toFixed(2)}%</Metric>
          </Card>
        </div>

        {/* Per-Class Classification Report */}
        <Card className="border border-slate-200 bg-white shadow-sm p-0 overflow-hidden">
          <div className="p-6 border-b border-slate-200">
            <Title className="text-slate-900">Per-Class Classification Report</Title>
            <Subtitle className="text-slate-500">Precision, recall, and F1-score for each attack taxonomy class.</Subtitle>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-[11px] uppercase bg-slate-50 text-slate-500 border-b border-slate-200">
                <tr>
                  <th className="px-6 py-3 text-left font-semibold">Class</th>
                  <th className="px-4 py-3 text-right font-semibold">Precision</th>
                  <th className="px-4 py-3 text-right font-semibold">Recall</th>
                  <th className="px-4 py-3 text-right font-semibold">F1-Score</th>
                  <th className="px-4 py-3 text-right font-semibold">Support</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {classLabels.map((cls) => {
                  const m = evalData.classification_report[cls];
                  const isNormal = cls === "normal";
                  return (
                    <tr key={cls} className={`hover:bg-slate-50 ${isNormal ? "bg-emerald-50/50" : ""}`}>
                      <td className="px-6 py-3 font-mono font-semibold text-slate-800">
                        <span className="flex items-center gap-2">
                          {isNormal
                            ? <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
                            : <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
                          }
                          {cls}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-slate-700">{(m.precision * 100).toFixed(2)}%</td>
                      <td className="px-4 py-3 text-right font-mono text-slate-700">{(m.recall * 100).toFixed(2)}%</td>
                      <td className="px-4 py-3 text-right font-mono font-bold text-blue-600">{(m.f1_score * 100).toFixed(2)}%</td>
                      <td className="px-4 py-3 text-right font-mono text-slate-500">{m.support.toLocaleString()}</td>
                    </tr>
                  );
                })}
                {/* Averages */}
                <tr className="bg-slate-50 border-t-2 border-slate-200">
                  <td className="px-6 py-3 font-semibold text-slate-850">Macro Avg</td>
                  <td className="px-4 py-3 text-right font-mono text-slate-700">{(evalData.macro_avg.precision * 100).toFixed(2)}%</td>
                  <td className="px-4 py-3 text-right font-mono text-slate-700">{(evalData.macro_avg.recall * 100).toFixed(2)}%</td>
                  <td className="px-4 py-3 text-right font-mono font-bold text-blue-600">{(evalData.macro_avg.f1_score * 100).toFixed(2)}%</td>
                  <td className="px-4 py-3 text-right text-slate-500">—</td>
                </tr>
                <tr className="bg-slate-50">
                  <td className="px-6 py-3 font-semibold text-slate-850">Weighted Avg</td>
                  <td className="px-4 py-3 text-right font-mono text-slate-700">{(evalData.weighted_avg.precision * 100).toFixed(2)}%</td>
                  <td className="px-4 py-3 text-right font-mono text-slate-700">{(evalData.weighted_avg.recall * 100).toFixed(2)}%</td>
                  <td className="px-4 py-3 text-right font-mono font-bold text-blue-600">{(evalData.weighted_avg.f1_score * 100).toFixed(2)}%</td>
                  <td className="px-4 py-3 text-right text-slate-500">—</td>
                </tr>
              </tbody>
            </table>
          </div>
        </Card>

        {/* Binary Detection Stats + Confusion Matrix */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Binary Detection Metrics */}
          <Card className="border border-slate-200 bg-white shadow-sm">
            <Title className="text-slate-900 mb-4">Binary Anomaly Detection</Title>
            <Subtitle className="text-slate-500 mb-4">Normal vs. Any Anomaly</Subtitle>
            <div className="space-y-3">
              {[
                { label: "True Positives", value: bd.true_positives, color: "text-emerald-600" },
                { label: "True Negatives", value: bd.true_negatives, color: "text-emerald-600" },
                { label: "False Positives", value: bd.false_positives, color: "text-rose-600" },
                { label: "False Negatives", value: bd.false_negatives, color: "text-rose-600" },
              ].map(({ label, value, color }) => (
                <div key={label} className="flex items-center justify-between py-2 border-b border-slate-200">
                  <span className="text-sm text-slate-600">{label}</span>
                  <span className={`font-mono font-bold text-sm ${color}`}>{value.toLocaleString()}</span>
                </div>
              ))}
              <div className="flex items-center justify-between py-2 mt-2">
                <span className="text-sm font-semibold text-slate-700">False Positive Rate</span>
                <span className="font-mono font-bold text-lg text-amber-600">{(bd.false_positive_rate * 100).toFixed(4)}%</span>
              </div>
              <div className="flex items-center justify-between py-2">
                <span className="text-sm font-semibold text-slate-700">Detection Rate (TPR)</span>
                <span className="font-mono font-bold text-lg text-blue-600">{(bd.detection_rate * 100).toFixed(2)}%</span>
              </div>
            </div>
          </Card>

          {/* Confusion Matrix */}
          <Card className="border border-slate-200 bg-white shadow-sm">
            <Title className="text-slate-900 mb-4">Confusion Matrix</Title>
            <Subtitle className="text-slate-500 mb-4">Predicted vs. Actual (all classes)</Subtitle>
            <div className="overflow-x-auto">
              <table className="text-[10px] font-mono w-full">
                <thead>
                  <tr>
                    <th className="px-1 py-1 text-slate-500 text-left">Actual / Pred</th>
                    {cm.labels.map((l) => (
                      <th key={l} className="px-1 py-1 text-slate-600 text-center whitespace-nowrap" style={{writingMode: "vertical-rl", maxWidth: "40px"}}>
                        {l.length > 10 ? l.slice(0, 10) + "…" : l}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {cm.matrix.map((row: number[], i: number) => {
                    const rowMax = Math.max(...row);
                    return (
                      <tr key={i}>
                        <td className="px-1 py-1 text-slate-600 whitespace-nowrap font-semibold">{cm.labels[i].length > 12 ? cm.labels[i].slice(0, 12) + "…" : cm.labels[i]}</td>
                        {row.map((val: number, j: number) => {
                          const isDiag = i === j;
                          const intensity = rowMax > 0 ? val / rowMax : 0;
                          return (
                            <td
                              key={j}
                              className={`px-1 py-1 text-center font-bold ${
                                isDiag
                                  ? "text-emerald-700"
                                  : val > 0
                                    ? "text-rose-600"
                                    : "text-slate-400"
                              }`}
                              style={{
                                backgroundColor: isDiag
                                  ? `rgba(16, 185, 129, ${intensity * 0.2})`
                                  : val > 0
                                    ? `rgba(244, 63, 94, ${intensity * 0.15})`
                                    : "transparent",
                              }}
                            >
                              {val}
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      </div>
    );
  };

  // ════════════════════════════════════════════════════════════════════
  // Tab: System Info
  // ════════════════════════════════════════════════════════════════════
  const SystemInfoTab = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <Card className="border border-slate-200 bg-white shadow-sm">
        <Text className="text-slate-500 font-medium flex items-center gap-2">
          <Zap className="w-4 h-4 text-blue-600" />
          API Gateway
        </Text>
        <Metric className="text-slate-950 mt-2 text-xl">FastAPI (Online)</Metric>
      </Card>
      <Card className="border border-slate-200 bg-white shadow-sm">
        <Text className="text-slate-500 font-medium flex items-center gap-2">
          <Activity className="w-4 h-4 text-blue-600" />
          Detection Engine
        </Text>
        <Metric className="text-slate-950 mt-2 text-xl">LSTM Autoencoder (PyTorch)</Metric>
      </Card>
      <Card className="border border-slate-200 bg-white shadow-sm">
        <Text className="text-slate-500 font-medium flex items-center gap-2">
          <Target className="w-4 h-4 text-blue-600" />
          Classification Engine
        </Text>
        <Metric className="text-slate-950 mt-2 text-xl">XGBoost Multi-Class Classifier</Metric>
      </Card>
      <Card className="border border-slate-200 bg-white shadow-sm">
        <Text className="text-slate-500 font-medium flex items-center gap-2">
          <Clock className="w-4 h-4 text-blue-600" />
          Frontend Framework
        </Text>
        <Metric className="text-slate-950 mt-2 text-xl">Next.js App Router</Metric>
      </Card>
      <Card className="border border-slate-200 bg-white shadow-sm md:col-span-2">
        <Text className="text-slate-500 font-medium flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-blue-600" />
          Explainability Layer
        </Text>
        <Metric className="text-slate-950 mt-2 text-xl">SHAP TreeExplainer + Human-Readable Attribution</Metric>
      </Card>
    </div>
  );

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Title */}
      <div>
        <Title className="text-2xl font-semibold text-slate-900">Anomaly Detection Assessment Report</Title>
        <Subtitle className="text-slate-500 mt-1">Autonomous monitoring for lateral movement, credential attacks, and behavioral anomalies.</Subtitle>
      </div>

      {activeTab === "Dashboard" && <DashboardTab />}
      {activeTab === "Threat Assessments" && <ThreatAssessmentsTab />}
      {activeTab === "Incident Reports" && <IncidentReportsTab />}
      {activeTab === "Evaluation Metrics" && <EvaluationMetricsTab />}
      {activeTab === "System Info" && <SystemInfoTab />}

    </div>
  );
}
