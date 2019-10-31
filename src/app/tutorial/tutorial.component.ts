import { Component, OnInit, ViewEncapsulation } from '@angular/core';
import { HttpClient } from '@angular/common/http';

import { Tutorial } from '../tutorial';

@Component({
  selector: 'app-tutorial',
  templateUrl: './tutorial.component.html',
  styleUrls: ['./tutorial.component.css'],
  encapsulation: ViewEncapsulation.None
})
export class TutorialComponent implements OnInit {
  selectedTutorial: string;
  currentSelection: string;
  currentTutorialText: string;
  tutorialList: Tutorial[];

  constructor(private http: HttpClient) { }

  ngOnInit() {
    this.getTutorialStructure();
    this.selectedTutorial = 't01_basic_usage.md';
    this.currentSelection = 'assets/tutorial/t01_basic_usage.md';
    this.getSelectedTutorialText();
  }

  getTutorialStructure() {
    this.http.get('assets/tutorial/structure.json', {responseType: 'text'}).subscribe(data => {
      this.tutorialList = <Tutorial[]>JSON.parse(data);
    });
  }

  updateCurrentTutorial(tutorial: Tutorial) {
    window.scroll(0,0);
    this.selectedTutorial = tutorial.name;
    this.currentSelection = 'assets/tutorial/' + this.selectedTutorial;

    this.getSelectedTutorialText();

  }

  getSelectedTutorialText() {
    this.http.get(this.currentSelection, {responseType: 'text'}).subscribe(data => {
      this.currentTutorialText = data;
    });
  }

}
