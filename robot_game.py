# -*- coding: utf-8 -*-
"""
==================================================
  Turn-Based Robot Battle Game
  Object-Oriented Programming Example
==================================================

Teams:
  Red  Team : Robot1 (Attacker/Defender), Robot2 (Healer)
  Blue Team : Robot3 (Attacker/Defender), Robot4 (Healer)

Rules:
  - 3 turns total (game may end early if a team is wiped out)
  - Each turn, the player chooses actions for every robot
  - Attacker robots : attack / defend
  - Healer robots   : heal (always targets the ally with the lowest HP)
  - Defense reduces incoming damage by the robot's defense value for that turn
  - Heal restores a fixed amount of HP (capped at max HP)
==================================================
"""

import sys
import io

# Windows 콘솔 UTF-8 출력 강제 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdin  = io.TextIOWrapper(sys.stdin.buffer,  encoding='utf-8', errors='replace')


# ──────────────────────────────────────────────
# Base Class
# ──────────────────────────────────────────────

class Robot:
    """모든 로봇의 기본 클래스"""

    def __init__(self, name: str, team: str, hp: int, max_hp: int):
        self.name = name
        self.team = team
        self.hp = hp
        self.max_hp = max_hp
        self.is_defending = False   # 해당 턴에 방어 중인지 여부

    @property
    def is_alive(self) -> bool:
        return self.hp > 0

    def take_damage(self, damage: int) -> int:
        """데미지를 받는다. 방어 중이면 방어력만큼 감소. 실제 받은 데미지를 반환."""
        if self.is_defending:
            reduction = self.defense_value()
            damage = max(0, damage - reduction)
        actual = min(damage, self.hp)
        self.hp -= actual
        return actual

    def defense_value(self) -> int:
        """하위 클래스에서 방어력 값을 반환하도록 오버라이드."""
        return 0

    def reset_turn_state(self):
        """매 턴 시작 시 상태 초기화"""
        self.is_defending = False

    def status(self) -> str:
        bar_len = 20
        filled = int(bar_len * self.hp / self.max_hp) if self.max_hp > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        return (f"  [{self.name}] HP: {self.hp}/{self.max_hp}  "
                f"|{bar}|  {'(ALIVE)' if self.is_alive else '(DEAD)'}")


# ──────────────────────────────────────────────
# Warrior Robot (공격 + 방어)
# ──────────────────────────────────────────────

class WarriorRobot(Robot):
    """공격과 방어가 가능한 전투 로봇"""

    ACTIONS = ["attack", "defend"]

    def __init__(self, name: str, team: str, hp: int, attack_power: int, defense_power: int):
        super().__init__(name, team, hp, max_hp=hp)
        self.attack_power = attack_power
        self.defense_power = defense_power

    def defense_value(self) -> int:
        return self.defense_power

    def attack(self, target: "Robot") -> str:
        if not self.is_alive:
            return f"  {self.name} 은(는) 이미 파괴되어 행동할 수 없습니다."
        damage = self.attack_power
        actual = target.take_damage(damage)
        msg = f"  ⚔  {self.name} → {target.name}: {damage} 데미지 시도"
        if target.is_defending:
            msg += f" (방어로 {target.defense_value()} 감소) → 실제 {actual} 데미지"
        else:
            msg += f" → 실제 {actual} 데미지"
        if not target.is_alive:
            msg += f"  💥 {target.name} 파괴됨!"
        return msg

    def defend(self) -> str:
        if not self.is_alive:
            return f"  {self.name} 은(는) 이미 파괴되어 행동할 수 없습니다."
        self.is_defending = True
        return f"  🛡  {self.name}: 방어 자세 취함 (다음 피격 시 {self.defense_power} 데미지 감소)"

    def info(self) -> str:
        return (f"{self.name} | 팀: {self.team} | "
                f"HP: {self.hp}/{self.max_hp} | "
                f"공격력: {self.attack_power} | 방어력: {self.defense_power}")


# ──────────────────────────────────────────────
# Healer Robot (힐 전용)
# ──────────────────────────────────────────────

class HealerRobot(Robot):
    """아군을 치료하는 힐러 로봇"""

    ACTIONS = ["heal"]
    HEAL_AMOUNT = 30   # 회복량

    def __init__(self, name: str, team: str, hp: int):
        super().__init__(name, team, hp, max_hp=hp)

    def heal(self, allies: list) -> str:
        if not self.is_alive:
            return f"  {self.name} 은(는) 이미 파괴되어 행동할 수 없습니다."
        # 살아있는 아군 중 HP가 가장 낮은 대상 선택
        alive_allies = [r for r in allies if r.is_alive and r is not self]
        if not alive_allies:
            # 아군이 없으면 자기 자신 힐
            alive_allies = [self] if self.is_alive else []
        if not alive_allies:
            return f"  {self.name}: 힐 대상 없음"
        target = min(alive_allies, key=lambda r: r.hp)
        before = target.hp
        target.hp = min(target.hp + self.HEAL_AMOUNT, target.max_hp)
        actual = target.hp - before
        return (f"  💚 {self.name} → {target.name} 힐: +{actual} HP "
                f"({before} → {target.hp})")

    def info(self) -> str:
        return (f"{self.name} | 팀: {self.team} | "
                f"HP: {self.hp}/{self.max_hp} | "
                f"회복량: {self.HEAL_AMOUNT} (자동 최저 HP 동료 대상)")


# ──────────────────────────────────────────────
# Game Manager
# ──────────────────────────────────────────────

class Game:
    MAX_TURNS = 3

    def __init__(self):
        # ── 로봇 생성 ──────────────────────────
        self.robot1 = WarriorRobot("Robot1", "Red",  hp=120, attack_power=35, defense_power=20)
        self.robot2 = HealerRobot ("Robot2", "Red",  hp=80)
        self.robot3 = WarriorRobot("Robot3", "Blue", hp=110, attack_power=40, defense_power=15)
        self.robot4 = HealerRobot ("Robot4", "Blue", hp=90)

        self.red_team  = [self.robot1, self.robot2]
        self.blue_team = [self.robot3, self.robot4]
        self.all_robots = self.red_team + self.blue_team

        self.turn = 1
        self.first_team: str = ""  # "Red" or "Blue"

    # ── 유틸리티 ──────────────────────────────

    def get_input(self, prompt: str) -> str:
        val = input(prompt).strip()
        if val.lower() in ('q', 'quit', 'exit'):
            print("\n  [게임을 중단하고 종료합니다.]")
            sys.exit(0)
        return val

    def divider(self, char="=", width=56):
        print(char * width)

    def print_all_status(self):
        print("\n  ── Red Team ──")
        for r in self.red_team:
            print(r.status())
        print("  ── Blue Team ──")
        for r in self.blue_team:
            print(r.status())
        print()

    def team_alive(self, team: list) -> bool:
        return any(r.is_alive for r in team)

    def get_alive_enemies(self, attacker_team: str) -> list:
        enemies = self.blue_team if attacker_team == "Red" else self.red_team
        return [r for r in enemies if r.is_alive]

    def get_alive_allies(self, team: str) -> list:
        allies = self.red_team if team == "Red" else self.blue_team
        return [r for r in allies if r.is_alive]

    # ── 입력 처리 ──────────────────────────────

    def choose_team_order(self):
        self.divider()
        print("  [ROBOT BATTLE GAME]")
        self.divider()
        print("\n  [로봇 정보]")
        for r in self.all_robots:
            print(f"    • {r.info()}")
        print()
        while True:
            choice = self.get_input("  먼저 행동할 팀을 선택하세요 (Red / Blue) [종료: q]: ").capitalize()
            if choice in ("Red", "Blue"):
                self.first_team = choice
                break
            print("  ✗ 'Red' 또는 'Blue'를 입력하세요.")

    def get_attack_target(self, attacker: WarriorRobot) -> Robot | None:
        enemies = self.get_alive_enemies(attacker.team)
        if not enemies:
            return None
        print(f"\n    공격 대상을 선택하세요 (적 팀 생존 로봇):")
        for i, e in enumerate(enemies, 1):
            print(f"      {i}. {e.name}  (HP: {e.hp}/{e.max_hp})")
        while True:
            try:
                idx = int(self.get_input(f"    번호 입력 (1~{len(enemies)}) [종료: q]: ")) - 1
                if 0 <= idx < len(enemies):
                    return enemies[idx]
            except ValueError:
                pass
            print("    ✗ 올바른 번호를 입력하세요.")

    def process_robot_action(self, robot: Robot):
        """한 로봇의 행동을 입력받아 실행"""
        if not robot.is_alive:
            print(f"\n  [{robot.name}] 이미 파괴됨 → 행동 건너뜀")
            return

        if isinstance(robot, WarriorRobot):
            print(f"\n  [{robot.name}] 행동 선택: attack(공격) / defend(방어)")
            while True:
                action = self.get_input(f"    선택 (종료: q) > ").lower()
                if action in WarriorRobot.ACTIONS:
                    break
                print("    ✗ 'attack' 또는 'defend'를 입력하세요.")
            if action == "attack":
                target = self.get_attack_target(robot)
                if target:
                    print(robot.attack(target))
                else:
                    print(f"  {robot.name}: 공격 대상 없음")
            else:
                print(robot.defend())

        elif isinstance(robot, HealerRobot):
            allies = (self.red_team if robot.team == "Red" else self.blue_team)
            print(f"\n  [{robot.name}] 자동으로 최저 HP 아군을 힐합니다.")
            print(robot.heal(allies))

    def process_team_turn(self, team_name: str):
        team = self.red_team if team_name == "Red" else self.blue_team
        color = "[Red]" if team_name == "Red" else "[Blue]"
        print(f"\n  {color} {team_name} Team 행동 단계")
        self.divider("-", 40)
        for robot in team:
            self.process_robot_action(robot)

    # ── 게임 루프 ─────────────────────────────

    def run_turn(self):
        self.divider()
        print(f"  ★ TURN {self.turn} / {self.MAX_TURNS}")
        self.divider()
        self.print_all_status()

        # 매 턴 시작 시 모든 로봇의 방어 상태 초기화
        for r in self.all_robots:
            r.reset_turn_state()

        second_team = "Blue" if self.first_team == "Red" else "Red"
        for team in [self.first_team, second_team]:
            self.process_team_turn(team)
            # 한 팀 행동 후 즉시 종료 조건 확인
            if not self.team_alive(self.red_team) or not self.team_alive(self.blue_team):
                return

        self.turn += 1

    def check_game_over(self) -> str | None:
        """게임 종료 조건 확인. 승리 팀 이름 반환, 진행 중이면 None."""
        red_alive  = self.team_alive(self.red_team)
        blue_alive = self.team_alive(self.blue_team)
        if not red_alive and not blue_alive:
            return "Draw"
        if not red_alive:
            return "Blue"
        if not blue_alive:
            return "Red"
        return None

    def show_result(self, winner: str):
        self.divider()
        if winner == "Draw":
            print("  [무승부] 모든 로봇이 파괴되었습니다.")
        else:
            loser = "Blue" if winner == "Red" else "Red"
            print(f"  [승리] {winner} Team 승리! ({loser} Team 전멸)")
        self.divider()
        print(f"\n  [승리 팀 최종 Status]")
        win_team = self.red_team if winner == "Red" else self.blue_team
        if winner == "Draw":
            win_team = self.all_robots
        for r in win_team:
            print(r.status())
        print()

    def run(self):
        self.choose_team_order()

        while self.turn <= self.MAX_TURNS:
            self.run_turn()
            winner = self.check_game_over()
            if winner is not None:
                self.show_result(winner)
                return

        # 3턴 종료 후 HP 합산으로 승패 결정
        self.divider()
        print(f"\n  ⏱  {self.MAX_TURNS}턴이 모두 종료되었습니다!")
        self.print_all_status()

        red_hp  = sum(r.hp for r in self.red_team)
        blue_hp = sum(r.hp for r in self.blue_team)
        print(f"  Red Team 잔여 총 HP: {red_hp}")
        print(f"  Blue Team 잔여 총 HP: {blue_hp}")

        if red_hp > blue_hp:
            winner = "Red"
        elif blue_hp > red_hp:
            winner = "Blue"
        else:
            winner = "Draw"

        self.show_result(winner)


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    game = Game()
    game.run()
