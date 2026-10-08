import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Component, type ErrorInfo, type ReactNode } from "react";
import { ApiProblem } from "../api/problem";
import { ProblemFromError } from "../components/ProblemView";

export function makeQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        // A 4xx will not change by asking again.
        retry: (count, error) => !(error instanceof ApiProblem && error.status < 500) && count < 2,
        refetchOnWindowFocus: false,
      },
    },
  });
}

type BoundaryState = { error: unknown; failed: boolean };

export class ErrorBoundary extends Component<{ children: ReactNode }, BoundaryState> {
  state: BoundaryState = { error: null, failed: false };

  static getDerivedStateFromError(error: unknown): BoundaryState {
    return { error, failed: true };
  }

  componentDidCatch(error: unknown, info: ErrorInfo): void {
    console.error(error, info.componentStack);
  }

  render() {
    if (this.state.failed) return <ProblemFromError error={this.state.error} />;
    return this.props.children;
  }
}

export function Providers({ children, client }: { children: ReactNode; client?: QueryClient }) {
  const qc = client ?? makeQueryClient();
  return (
    <QueryClientProvider client={qc}>
      <ErrorBoundary>{children}</ErrorBoundary>
    </QueryClientProvider>
  );
}
