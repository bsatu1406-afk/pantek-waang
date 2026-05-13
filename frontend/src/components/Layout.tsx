import {
  Activity,
  Database,
  KeyRound,
  LayoutDashboard,
  LogOut,
  Radio,
  Server,
  ServerCog,
} from "lucide-react";
import { type ReactNode } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "@/lib/AuthContext";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/live", label: "Live", icon: Radio },
  { to: "/data-inspector", label: "Data Inspector", icon: Database },
  { to: "/api-keys", label: "API Keys", icon: KeyRound },
  { to: "/databento-keys", label: "Databento Keys", icon: Server },
  { to: "/system-status", label: "System Status", icon: ServerCog },
];

export function Layout({ children }: { children: ReactNode }) {
  const { logout } = useAuth();
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="flex">
        <aside className="hidden w-60 shrink-0 border-r border-border bg-background/60 p-4 md:block">
          <div className="mb-6 flex items-center gap-2 px-2">
            <Activity className="h-5 w-5 text-primary" />
            <span className="text-base font-semibold tracking-tight">Options Flow</span>
          </div>
          <nav className="space-y-1">
            {NAV.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) =>
                    cn(
                      "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                      isActive
                        ? "bg-accent text-accent-foreground"
                        : "text-muted-foreground hover:bg-accent/40 hover:text-foreground",
                    )
                  }
                >
                  <Icon className="h-4 w-4" />
                  {item.label}
                </NavLink>
              );
            })}
          </nav>
          <button
            onClick={() => {
              logout();
              navigate("/login");
            }}
            className="mt-8 flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent/40 hover:text-foreground"
          >
            <LogOut className="h-4 w-4" />
            Sign out
          </button>
        </aside>
        <main className="flex-1 px-6 py-6">{children}</main>
      </div>
    </div>
  );
}
