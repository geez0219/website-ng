import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { SlackFormComponent } from './slack-form.component';

describe('SlackFormComponent', () => {
  let component: SlackFormComponent;
  let fixture: ComponentFixture<SlackFormComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ SlackFormComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(SlackFormComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
