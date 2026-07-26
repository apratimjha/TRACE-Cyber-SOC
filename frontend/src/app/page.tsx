"use client";

import { useState } from "react";
import Dashboard from "@/components/Dashboard";
import { LayoutDashboard, ShieldAlert, FileText, Server, BarChart3 } from "lucide-react";

export default function Home() {
  const [activeTab, setActiveTab] = useState("Dashboard");
  const [severityFilter, setSeverityFilter] = useState("all");

  const tabs = [
    { name: "Dashboard", icon: LayoutDashboard },
    { name: "Threat Assessments", icon: ShieldAlert },
    { name: "Incident Reports", icon: FileText },
    { name: "Evaluation Metrics", icon: BarChart3 },
    { name: "System Info", icon: Server },
  ];

  const severityOptions = [
    { value: "all", label: "All Severities" },
    { value: "critical", label: "Critical (>0.8)" },
    { value: "high", label: "High (0.6–0.8)" },
    { value: "medium", label: "Medium (0.4–0.6)" },
  ];

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50 text-slate-900 font-sans">
      {/* Light Left Sidebar */}
      <aside className="w-64 bg-white border-r border-slate-200 text-slate-600 flex flex-col justify-between shrink-0 h-full">
        <div>
          <div className="p-6">
            <h1 className="text-xl font-bold tracking-widest text-slate-900 flex items-center gap-2">
              <ShieldAlert className="w-6 h-6 text-blue-600" />
              T.R.A.C.E.
            </h1>
            <p className="text-[10px] text-slate-500 mt-1 font-medium tracking-wide leading-tight">
              Temporal Recognition of Anomalous Cyber Events
            </p>
          </div>
          <nav className="mt-4 flex flex-col gap-1 px-3">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.name;
              return (
                <button
                  key={tab.name}
                  onClick={() => setActiveTab(tab.name)}
                  className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all w-full text-left text-sm ${
                    isActive
                      ? "bg-blue-50 text-blue-600 border-l-[3px] border-blue-600"
                      : "border-l-[3px] border-transparent hover:bg-slate-100 hover:text-slate-900 cursor-pointer"
                  }`}
                >
                  <Icon className="w-[18px] h-[18px]" />
                  <span className="font-medium">{tab.name}</span>
                </button>
              );
            })}
          </nav>

          {/* Severity Filter — visible when on Threat Assessments tab */}
          {activeTab === "Threat Assessments" && (
            <div className="mt-6 px-4">
              <label className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold mb-2 block">
                Severity Filter
              </label>
              <div className="flex flex-col gap-1">
                {severityOptions.map((opt) => (
                  <button
                    key={opt.value}
                    onClick={() => setSeverityFilter(opt.value)}
                    className={`text-xs px-3 py-2 rounded-md text-left transition-all ${
                      severityFilter === opt.value
                        ? "bg-blue-50 text-blue-600 font-semibold"
                        : "text-slate-500 hover:bg-slate-100 hover:text-slate-900"
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
        
        <div className="p-4 text-xs text-slate-400 border-t border-slate-200">
          Built by Apratim K Jha
        </div>
      </aside>

      {/* Main Content Area — Light */}
      <main className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Top Header */}
        <header className="bg-white border-b border-slate-200 h-14 flex items-center px-8 shrink-0">
          <div className="text-sm font-medium text-blue-600">
            Reports <span className="text-slate-300 mx-2">/</span> <span className="text-slate-600">{activeTab}</span>
          </div>
        </header>

        {/* Dashboard Content */}
        <div className="flex-1 overflow-auto p-8 bg-slate-50">
          <Dashboard activeTab={activeTab} severityFilter={severityFilter} />
        </div>
      </main>
    </div>
  );
}
