import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { useAuth } from "./lib/AuthContext";
import { ApiKeysPage } from "./pages/ApiKeys";
import { DashboardPage } from "./pages/Dashboard";
import { DataInspectorPage } from "./pages/DataInspector";
import { LivePage } from "./pages/Live";
import { LoginPage } from "./pages/Login";
import { SystemStatusPage } from "./pages/SystemStatus";

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { token } = useAuth();
  if (!token) return <Navigate to="/login" replace />;
  return <Layout>{children}</Layout>;
}

export function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <DashboardPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/api-keys"
        element={
          <ProtectedRoute>
            <ApiKeysPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/system-status"
        element={
          <ProtectedRoute>
            <SystemStatusPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/data-inspector"
        element={
          <ProtectedRoute>
            <DataInspectorPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/live"
        element={
          <ProtectedRoute>
            <LivePage />
          </ProtectedRoute>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
