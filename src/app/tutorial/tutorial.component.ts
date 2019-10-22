import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http'; 
import { Tutorial } from '../tutorial';

@Component({
  selector: 'app-tutorial',
  templateUrl: './tutorial.component.html',
  styleUrls: ['./tutorial.component.css']
})
export class TutorialComponent implements OnInit {
  apiList: string[];
  currentSelection: string;
  currentTutorialText: string;
  tutorialList: Tutorial[];

  constructor(private http: HttpClient) { }

  ngOnInit() {
    this.getTutorialSummary();
    this.currentSelection = 'assets/tutorial/t01_basic_usage.md';
    this.getSelectedTutorialText();
  }

  getTutorialSummary() {
    this.http.get('assets/tutorial/structure.json', {responseType: 'text'}).subscribe(data => {
      this.tutorialList = <Tutorial[]>JSON.parse(data);

      console.log(this.tutorialList);
    });
  }
  updateCurrentTutorial(tutorial: Tutorial) {
    this.currentSelection = 'assets/tutorial/' + tutorial.name;

    this.getSelectedTutorialText();
  }

  getSelectedTutorialText() {
    this.http.get(this.currentSelection, {responseType: 'text'}).subscribe(data => {
      this.currentTutorialText = data;
    });
  }

}
