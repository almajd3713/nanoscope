import { render, screen, act } from "@testing-library/react";
import { useQuery } from "@tanstack/react-query";
import { afterEach, describe, expect, it } from "vitest";
import { ApiProblem } from "../api/problem";
import { resetAuth } from "../app/auth";
import { Providers } from "../app/providers";
import { Login } from "./Login";

afterEach(() => act(() => resetAuth()));

describe("Login", () => {
  it("says where to find the login link", () => {
    render(<Login />);
    expect(screen.getByRole("heading", { name: "Sign in with the login link" })).toBeTruthy();
    expect(screen.getByText("docker compose logs api | grep login")).toBeTruthy();
    expect(screen.getByText("~/.nanoscope/server/token")).toBeTruthy();
    expect(screen.getByText(/run them as your user/)).toBeTruthy();
  });

  it("replaces the screen when a request comes back 401", async () => {
    function Screen() {
      useQuery({
        queryKey: ["x"],
        queryFn: () => {
          throw new ApiProblem(401, { title: "Not signed in", detail: "sign in with the token" });
        },
      });
      return <p>the screen</p>;
    }
    render(
      <Providers>
        <Screen />
      </Providers>,
    );
    expect(await screen.findByRole("heading", { name: "Sign in with the login link" })).toBeTruthy();
    expect(screen.queryByText("the screen")).toBeNull();
  });
});
