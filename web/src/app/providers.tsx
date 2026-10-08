import { MutationCache, QueryCache, QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Component, type ErrorInfo, type ReactNode, useState } from "react";
import { ApiProblem } from "../api/problem";
import { ProblemFromError } from "../components/ProblemView";
import { Login } from "../pages/Login";
import { markSignedOut, useSignedOut } from "./auth";

function on401(error: unknown): void {
  if (error instanceof ApiProblem && error.status === 401) markSignedOut();
}

export function makeQueryClient(): QueryClient {
  return new QueryClient({
    queryCache: new QueryCache({ onError: on401 }),
    mutationCache: new MutationCache({ onError: on401 }),
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

function Gate({ children }: { children: ReactNode }) {
  return useSignedOut() ? <Login /> : children;
}

export function Providers({ children, client }: { children: ReactNode; client?: QueryClient }) {
  const [fallback] = useState(makeQueryClient);
  const qc = client ?? fallback;
  return (
    <QueryClientProvider client={qc}>
      <ErrorBoundary>
        <Gate>{children}</Gate>
      </ErrorBoundary>
    </QueryClientProvider>
  );
}
