const { useState, useEffect, useRef } = React;

const Card = ({ children, title, className = "" }) => (
  <div
    className={`glass rounded-2xl p-6 transition-all duration-300 ${className}`}
  >
    {title && (
      <h3 className="text-lg font-medium mb-4 text-slate-200 border-b border-slate-700/30 pb-3 flex items-center gap-2">
        {title}
      </h3>
    )}
    {children}
  </div>
);

const Button = ({
  onClick,
  children,
  variant = "primary",
  disabled = false,
  className = "",
}) => {
  const variants = {
    primary:
      "bg-cyan-500 hover:bg-cyan-400 text-slate-900 shadow-lg shadow-cyan-500/10",
    danger:
      "bg-rose-500 hover:bg-rose-400 text-white shadow-lg shadow-rose-500/10",
    secondary: "bg-slate-700 hover:bg-slate-600 text-slate-200",
  };
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`px-5 py-2.5 rounded-xl font-semibold text-sm transition-all active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed ${variants[variant]} ${className}`}
    >
      {children}
    </button>
  );
};

const StatusBadge = ({ status }) => {
  let colorClass = "bg-slate-500/10 text-slate-400 border-slate-500/20";
  let label = "未知";

  if (status === "running") {
    colorClass =
      "bg-emerald-500/10 text-emerald-400 border-emerald-500/20 shadow-sm shadow-emerald-500/5";
    label = "运行中";
  }
  if (status === "stopping") {
    colorClass = "bg-amber-500/10 text-amber-400 border-amber-500/20";
    label = "停止中";
  }
  if (status === "stopped") {
    label = "已停止";
  }
  if (status === "error") {
    colorClass = "bg-rose-500/10 text-rose-400 border-rose-500/20";
    label = "错误";
  }

  return (
    <span
      className={`px-2.5 py-1 rounded-lg text-xs font-medium border ${colorClass} inline-flex items-center gap-1.5`}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${status === "running" ? "bg-emerald-400 animate-pulse" : "bg-current"}`}
      ></span>
      {label}
    </span>
  );
};

// Hook for WebSocket
const useWebSocket = (url) => {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState("disconnected");
  const ws = useRef(null);

  useEffect(() => {
    const connect = () => {
      ws.current = new WebSocket(url);
      ws.current.onopen = () => setStatus("connected");
      ws.current.onmessage = (e) => setData(JSON.parse(e.data));
      ws.current.onclose = () => {
        setStatus("disconnected");
        setTimeout(connect, 3000); // Reconnect
      };
    };
    connect();
    return () => ws.current?.close();
  }, [url]);

  return { data, status };
};

const getBoardInfo = (symbol) => {
  if (!symbol) return null;

  if (symbol.startsWith("688")) {
    return {
      name: "科创板",
      color: "bg-purple-500/20 text-purple-400 border-purple-500/30",
      dotColor: "bg-purple-400",
    };
  } else if (symbol.startsWith("300") || symbol.startsWith("301")) {
    return {
      name: "创业板",
      color: "bg-orange-500/20 text-orange-400 border-orange-500/30",
      dotColor: "bg-orange-400",
    };
  } else if (symbol.startsWith("4") || symbol.startsWith("8")) {
    return {
      name: "北交所",
      color: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
      dotColor: "bg-cyan-400",
    };
  }

  return {
    name: "主板",
    color: "bg-blue-500/20 text-blue-400 border-blue-500/30",
    dotColor: "bg-blue-400",
  };
};

window.Card = Card;
window.Button = Button;
window.StatusBadge = StatusBadge;
window.useWebSocket = useWebSocket;
window.getBoardInfo = getBoardInfo;
